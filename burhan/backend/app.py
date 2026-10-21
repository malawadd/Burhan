import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import struct
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, load_settings
from .capabilities import available, capability_key, load_tasks, selection_key
from .models import CheckEvent, NewSession, Pair, Plan, ToolResults, Turn
from .providers import BedrockAgent, CodexAgent, OffAgent, ProviderFailure

LOG = logging.getLogger('burhan.smart')
MAX_BODY = 6_100_000
ACTIVE = ('queued', 'running')
bearer = HTTPBearer(auto_error=False)


class PlanRejection(ProviderFailure):
    def __init__(self, message, code, details):
        super().__init__(message)
        self.code, self.details = code, details


def uid():
    return secrets.token_urlsafe(18)


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def image_dimensions(raw):
    """Inspect dimensions before allocating decoded pixels, including compressed image bombs."""
    if raw.startswith(b'\x89PNG\r\n\x1a\n') and raw[12:16] == b'IHDR' and len(raw) >= 24:
        return struct.unpack('>II',raw[16:24])
    if raw.startswith(b'\xff\xd8'):
        offset = 2
        while offset + 4 <= len(raw):
            if raw[offset] != 0xff:
                break
            while offset < len(raw) and raw[offset] == 0xff:
                offset += 1
            marker = raw[offset]
            offset += 1
            if marker in (0xd9,0xda):
                break
            if marker == 0x01 or 0xd0 <= marker <= 0xd8:
                continue
            length = int.from_bytes(raw[offset:offset+2],'big')
            if length < 2 or offset + length > len(raw):
                break
            if marker in (0xc0,0xc1,0xc2,0xc3,0xc5,0xc6,0xc7,0xc9,0xca,0xcb,0xcd,0xce,0xcf):
                height,width = struct.unpack('>HH',raw[offset+3:offset+7])
                return width,height
            offset += length
    raise ValueError('Unsupported or malformed image header')


class Service:
    def __init__(self, settings, provider=None):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        for name in ('snapshots', 'work'):
            (settings.data_dir / name).mkdir(exist_ok=True)
        self.db = sqlite3.connect(settings.data_dir / 'sessions.sqlite3', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY, token_hash TEXT UNIQUE, name TEXT, expires REAL);
          CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, device TEXT, metadata TEXT, closed INTEGER DEFAULT 0,
            scene_version INTEGER DEFAULT -1, created REAL);
          CREATE TABLE IF NOT EXISTS turns(id TEXT PRIMARY KEY, session TEXT, device TEXT, request_id TEXT,
            scene_version INTEGER, status TEXT, provider TEXT, context TEXT, snapshot TEXT, result TEXT,
            error TEXT, created REAL, started REAL, finished REAL, metrics TEXT);
          CREATE UNIQUE INDEX IF NOT EXISTS turn_retry ON turns(session,request_id) WHERE request_id IS NOT NULL;
          CREATE INDEX IF NOT EXISTS turn_queue ON turns(status,created);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session TEXT,turn_id TEXT,
            kind TEXT,payload TEXT,created REAL);
        ''')
        # Additive migration keeps every existing session and trace readable.
        if 'client_event_id' not in {r['name'] for r in self.db.execute('PRAGMA table_info(events)')}:
            self.db.execute('ALTER TABLE events ADD COLUMN client_event_id TEXT')
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS check_event_retry ON events(session,client_event_id) WHERE client_event_id IS NOT NULL')
        self.db.execute("UPDATE turns SET status='cancelled',error='Backend restarted; request another view.',finished=? WHERE status IN ('running','queued')", (time.time(),))
        self.db.commit()
        self.provider = provider or {'codex': CodexAgent, 'aws': BedrockAgent, 'off': lambda _: OffAgent()}[settings.provider](settings)
        self.pair_code = secrets.token_hex(4).upper()
        self.pair_expires = time.time() + 3600
        pairing = settings.data_dir / 'pairing.json'
        pairing.write_text(dump({'code': self.pair_code, 'expiresAt': self.pair_expires}), encoding='utf-8')
        try:
            pairing.chmod(0o600)
        except OSError:
            pass
        self.pair_attempts = defaultdict(deque)
        self.signal = asyncio.Event()
        self.current_id = None
        self.current_task = None
        self.worker_task = None
        self.maintenance_task = None
        self.tasks = load_tasks(settings)
        self.registry = {t['id']: t for t in self.tasks}

    def allowed_registry(self, context):
        return {key: task for key, task in self.registry.items() if available(task, context)}

    def current_context(self, session):
        latest = self.db.execute('SELECT context FROM turns WHERE session=? ORDER BY created DESC LIMIT 1', (session['id'],)).fetchone()
        return json.loads(latest['context'] if latest else session['metadata'])

    def event(self, session, turn, kind, payload):
        self.db.execute('INSERT INTO events(session,turn_id,kind,payload,created) VALUES(?,?,?,?,?)',
                        (session, turn, kind, dump(payload), time.time()))
        self.db.commit()

    def session(self, sid, device, allow_closed=False):
        row = self.db.execute('SELECT * FROM sessions WHERE id=? AND device=?', (sid, device)).fetchone()
        if row is None:
            raise HTTPException(404, 'Inspection session not found')
        if row['closed'] and not allow_closed:
            raise HTTPException(410, 'Inspection session is closed')
        return row

    def turn(self, sid, tid):
        row = self.db.execute('SELECT * FROM turns WHERE id=? AND session=?', (tid, sid)).fetchone()
        if row is None:
            raise HTTPException(404, 'Inspection turn not found')
        return row

    def cancel(self, tid, status='superseded'):
        self.db.execute('UPDATE turns SET status=?,finished=? WHERE id=? AND status IN (?,?)',
                        (status, time.time(), tid, *ACTIVE))
        self.db.commit()
        if self.current_id == tid and self.current_task:
            self.current_task.cancel()

    def cleanup(self):
        cutoff = time.time() - 86400
        for path in (self.settings.data_dir / 'snapshots').glob('*.jpg'):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        self.db.execute('DELETE FROM devices WHERE expires<?', (time.time(),))
        self.db.commit()

    def snapshot(self, image, tid):
        if not image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,')):
            raise HTTPException(422, 'Snapshot must be a JPEG or PNG data URL')
        try:
            raw = base64.b64decode(image.split(',', 1)[1], validate=True)
            if len(raw) > 4_000_000:
                raise ValueError('Snapshot exceeds 4 MB')
            width,height = image_dimensions(raw)
            if min(width,height) < 32 or max(width,height) > 4096 or width * height > 12_000_000:
                raise ValueError('Snapshot dimensions outside limits')
            mat = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if mat is None or min(mat.shape[:2]) < 32 or max(mat.shape[:2]) > 4096 or mat.shape[0] * mat.shape[1] > 12_000_000:
                raise ValueError('Snapshot dimensions must be 32–4096 pixels, at most 12 megapixels')
            h, w = mat.shape[:2]
            if max(h, w) > 1280:
                mat = cv2.resize(mat, (round(w * 1280 / max(h,w)), round(h * 1280 / max(h,w))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
            quality = {'opencvVersion': cv2.__version__, 'width': mat.shape[1], 'height': mat.shape[0],
                'laplacianVariance': round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                'meanBrightness': round(float(gray.mean()), 2),
                'darkFraction': round(float(np.mean(gray < 25)), 4),
                'brightFraction': round(float(np.mean(gray > 240)), 4)}
            path = self.settings.data_dir / 'snapshots' / (tid + '.jpg')
            if not cv2.imwrite(str(path), mat, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                raise ValueError('Snapshot encoding failed')
            return str(path), quality
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, 'Invalid or oversized inspection snapshot') from exc

    def validate_plan(self, plan, context):
        # Registry validation happens for both providers, never trusting names or modes from a model.
        plan = plan.model_copy(deep=True)
        normalizations = []
        registry = self.allowed_registry(context)
        for proposal in plan.suggestions:
            if proposal.taskId not in registry:
                raise PlanRejection('The agent suggested an unavailable inspection check.', 'suggestion_capability_mismatch', {'taskId':proposal.taskId})
        for action in plan.actions:
            if action.taskId is not None and action.taskId not in registry:
                raise PlanRejection('The agent requested an unsupported inspection task.', 'unknown_action_task', {'actionId':action.id,'taskId':action.taskId})
        target = plan.guidanceTaskId
        if target is not None and target not in registry:
            raise PlanRejection('The agent addressed an unsupported check.','unknown_guidance_task',{'taskId':target})
        if context.get('selectionMode') == 'confirm':
            active = context.get('activeTaskId') or None
            if target is not None and target != active:
                raise PlanRejection('Guidance belongs to a different check. Request another view.','guidance_task_mismatch',{'taskId':target,'activeTaskId':active})
            # Older providers omit this optional association. Bind to the explicit input,
            # never to the model's suggested task or select_detector proposal.
            plan.guidanceTaskId = active
            suppressed = set(context.get('dismissedTaskIds',[])) | set(context.get('completedTaskIds',[]))
            plan.suggestions = [s for s in plan.suggestions if s.taskId not in suppressed and s.taskId != active]
            plan.actions = [a for a in plan.actions if a.type != 'select_detector' or a.taskId not in suppressed]
        for suggestion in plan.suggestions:
            task = registry.get(suggestion.taskId)
            if not task or task['mode'] != suggestion.mode or bool(task['automatic']) != suggestion.automatic:
                raise PlanRejection('The agent suggested an unsupported inspection capability.', 'suggestion_capability_mismatch',
                    {'taskId':suggestion.taskId,'mode':suggestion.mode,'automatic':suggestion.automatic,
                     'expected':{'mode':task['mode'],'automatic':task['automatic']} if task else None})
        if len({a.id for a in plan.actions}) != len(plan.actions):
            raise PlanRejection('The agent returned duplicate action identifiers.','duplicate_action_ids',{})
        for action in plan.actions:
            if action.taskId is not None:
                task = registry.get(action.taskId)
                if not task:
                    raise PlanRejection('The agent requested an unsupported inspection task.','unknown_action_task',
                        {'actionId':action.id,'taskId':action.taskId})
                if action.type == 'select_detector' and action.mode is not None and action.mode != task['mode']:
                    raise PlanRejection('The agent requested an unsupported inspection task.','detector_task_mode_mismatch',
                        {'actionId':action.id,'taskId':action.taskId,'mode':action.mode,'expectedMode':task['mode']})
            # Informational camera/manual actions do not select or run a detector. Some models
            # echo the currently installed detector here while naming a manual catalogue task.
            # Canonicalize that irrelevant field; never repair IDs or detector selections.
            if action.type != 'select_detector' and action.mode is not None:
                normalizations.append({'actionId':action.id,'field':'mode','previous':action.mode,'value':None})
                action.mode = None
            if context.get('selectionMode') != 'confirm' and context.get('locked') and (action.mode not in (None,context.get('mode')) or action.taskId not in (None,context.get('activeTaskId'))):
                raise PlanRejection('The agent tried to change a locked inspection task.','locked_task_change',
                    {'actionId':action.id,'taskId':action.taskId,'mode':action.mode,'activeTaskId':context.get('activeTaskId')})
            if action.type == 'select_detector':
                if action.mode not in ('crack', 'surface', 'ppe', 'research'):
                    raise PlanRejection('The agent requested an unavailable detector.','unavailable_detector',{'actionId':action.id,'mode':action.mode})
                if context.get('selectionMode') == 'confirm' and not action.taskId:
                    raise PlanRejection('A detector proposal must identify its check.','missing_detector_task',{'actionId':action.id})
                if action.mode == 'research' and (not action.taskId or not registry[action.taskId].get('modelId')):
                    raise PlanRejection('Expanded detector proposals must identify their registered check.','missing_detector_task',{'actionId':action.id})
        return plan,normalizations

    async def worker(self):
        while True:
            await self.signal.wait()
            self.signal.clear()
            self.cleanup()
            while True:
                row = self.db.execute("SELECT * FROM turns WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if row is None:
                    break
                tid = row['id']
                session = self.db.execute('SELECT * FROM sessions WHERE id=?', (row['session'],)).fetchone()
                if session['closed'] or row['scene_version'] != session['scene_version']:
                    self.cancel(tid)
                    continue
                self.current_id = tid
                self.db.execute("UPDATE turns SET status='running',started=? WHERE id=?", (time.time(), tid))
                self.db.commit()
                metadata = json.loads(session['metadata'])
                ctx = json.loads(row['context'])
                history = [dict(r) for r in self.db.execute(
                    "SELECT id,scene_version,result FROM turns WHERE session=? AND status='completed' ORDER BY created DESC LIMIT 5", (row['session'],)).fetchall()][::-1]
                for item in history:
                    item['result'] = json.loads(item['result'])
                events = [json.loads(r['payload']) for r in self.db.execute(
                    "SELECT payload FROM events WHERE session=? AND kind='tool_results' ORDER BY id DESC LIMIT 8", (row['session'],)).fetchall()][::-1]
                check_events = [json.loads(r['payload']) for r in self.db.execute(
                    "SELECT payload FROM events WHERE session=? AND kind='check_event' ORDER BY id DESC LIMIT 30", (row['session'],)).fetchall()][::-1]
                catalogue = [{k:t.get(k) for k in ('id','title','titleAr','stage','mode','automatic','requiredEvidence','family','categoryId','modelId','researchOnly','scope')}
                             for t in self.tasks if available(t, ctx)]
                provider_context = {'inspection': metadata, 'sceneVersion': row['scene_version'], 'context': ctx,
                    'recentPlans': history, 'toolResults': events, 'checkEvents':check_events, 'capabilityCatalogue': catalogue,
                    'qualityNote': 'Laplacian variance is a heuristic, not a universal threshold. Never infer defect absence from a poor image.'}
                self.event(row['session'], tid, 'perception', {'sceneVersion': row['scene_version'], 'context': ctx})
                start = time.monotonic()
                try:
                    self.current_task = asyncio.create_task(self.provider.plan(provider_context, Path(row['snapshot'])))
                    plan, metrics = await self.current_task
                    try:
                        plan,normalizations = self.validate_plan(plan, ctx)
                    except PlanRejection as rejected:
                        self.event(row['session'],tid,'rejected_plan',{'code':rejected.code,'details':rejected.details,'plan':plan.model_dump()})
                        raise
                    if normalizations:
                        self.event(row['session'],tid,'plan_normalized',{'changes':normalizations})
                    fresh = self.db.execute('SELECT scene_version,closed FROM sessions WHERE id=?', (row['session'],)).fetchone()
                    state = self.db.execute('SELECT status FROM turns WHERE id=?', (tid,)).fetchone()['status']
                    if fresh['closed'] or fresh['scene_version'] != row['scene_version'] or state != 'running':
                        self.cancel(tid)
                        continue
                    metrics['durationMs'] = round((time.monotonic() - start) * 1000)
                    self.db.execute("UPDATE turns SET status='completed',result=?,metrics=?,finished=? WHERE id=?", (plan.model_dump_json(), dump(metrics), time.time(), tid))
                    self.db.commit()
                    self.event(row['session'], tid, 'decision', {'plan': plan.model_dump(), 'metrics': metrics})
                except asyncio.CancelledError:
                    state = self.db.execute('SELECT status FROM turns WHERE id=?', (tid,)).fetchone()['status']
                    if state == 'running':
                        self.cancel(tid, 'cancelled')
                    if asyncio.current_task().cancelling():
                        raise
                except Exception as exc:
                    message = str(exc) if isinstance(exc, ProviderFailure) else ('Inspection planning timed out. Try another view.' if isinstance(exc, TimeoutError) else 'Inspection planning failed. Offline detection remains available.')
                    self.db.execute("UPDATE turns SET status='failed',error=?,finished=? WHERE id=? AND status='running'", (message, time.time(), tid))
                    self.db.commit()
                    self.event(row['session'], tid, 'failure', {'error': message})
                    LOG.warning('Inspection turn %s failed (%s)', tid, type(exc).__name__)
                finally:
                    self.current_id = self.current_task = None

    async def maintain(self):
        while True:
            await asyncio.sleep(60)
            self.cleanup()


def create_app(settings: Settings | None = None, provider=None):
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app):
        service = Service(settings, provider)
        app.state.service = service
        service.cleanup()
        service.worker_task = asyncio.create_task(service.worker())
        service.maintenance_task = asyncio.create_task(service.maintain())
        LOG.warning('Burhan pairing code: %s (valid one hour; local file data/pairing.json). Provider: %s', service.pair_code, settings.provider)
        try:
            yield
        finally:
            service.worker_task.cancel()
            service.maintenance_task.cancel()
            try:
                await service.worker_task
            except asyncio.CancelledError:
                pass
            try:
                await service.maintenance_task
            except asyncio.CancelledError:
                pass
            service.db.close()

    app = FastAPI(title='Burhan Smart Inspection', version='1.3.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=False,
                       allow_methods=['GET','POST','DELETE','OPTIONS'], allow_headers=['Authorization','Content-Type'])

    @app.middleware('http')
    async def body_limit(request, call_next):
        # Bound chunked bodies as well as Content-Length. Never log photographs or bearer tokens.
        if request.method == 'POST':
            length = request.headers.get('content-length')
            if length and (not length.isdigit() or int(length) > MAX_BODY):
                from starlette.responses import JSONResponse
                return JSONResponse({'detail': 'Request body is too large'}, status_code=413)
            size, chunks = 0, []
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY:
                    from starlette.responses import JSONResponse
                    return JSONResponse({'detail': 'Request body is too large'}, status_code=413)
                chunks.append(chunk)
            request._body = b''.join(chunks)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    async def auth(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None:
            raise HTTPException(401, 'Pair this device first', headers={'WWW-Authenticate':'Bearer'})
        hashed = hashlib.sha256(credentials.credentials.encode()).hexdigest()
        row = request.app.state.service.db.execute('SELECT id FROM devices WHERE token_hash=? AND expires>?', (hashed, time.time())).fetchone()
        if not row:
            raise HTTPException(401, 'Device token is invalid or expired')
        return row['id']

    @app.get('/api/v1/health')
    async def health(request: Request):
        s = request.app.state.service
        return {'status':'ok', 'provider':settings.provider, 'ready':s.provider.ready()}

    @app.post('/api/v1/pair')
    async def pair(body: Pair, request: Request):
        s = request.app.state.service
        # X-Forwarded-For is intentionally not trusted for rate limiting.
        peer = request.client.host if request.client else 'unknown'
        attempts = s.pair_attempts[peer]
        now = time.time()
        while attempts and attempts[0] < now - 60:
            attempts.popleft()
        if len(attempts) >= 5:
            raise HTTPException(429, 'Too many pairing attempts. Wait one minute.')
        attempts.append(now)
        if now > s.pair_expires or not hmac.compare_digest(body.code.strip().upper(), s.pair_code):
            raise HTTPException(401, 'Pairing code is invalid or expired. Obtain the current code from the backend operator.')
        token = secrets.token_urlsafe(48)
        s.db.execute('INSERT INTO devices VALUES(?,?,?,?)', (uid(), hashlib.sha256(token.encode()).hexdigest(), body.deviceName, now + 30 * 86400))
        s.db.commit()
        return {'token':token}

    @app.post('/api/v1/sessions', status_code=201)
    async def new_session(body: NewSession, request: Request, device=Depends(auth)):
        s = request.app.state.service
        registry = s.allowed_registry(body.model_dump())
        if body.taskId and body.taskId not in registry:
            raise HTTPException(422, 'Inspection check is unavailable for this device')
        if body.mode == 'research' and (not body.taskId or registry[body.taskId]['mode'] != 'research'):
            raise HTTPException(422, 'Expanded inspection mode requires a registered installed detector check')
        if s.db.execute('SELECT COUNT(*) FROM sessions WHERE device=? AND closed=0',(device,)).fetchone()[0] >= 20:
            raise HTTPException(429, 'Close an older inspection session before starting another.')
        sid = uid()
        s.db.execute('INSERT INTO sessions(id,device,metadata,created) VALUES(?,?,?,?)', (sid,device,body.model_dump_json(),time.time()))
        s.db.commit()
        return {'id':sid,'provider':settings.provider}

    @app.post('/api/v1/sessions/{sid}/turns', status_code=202)
    async def submit_turn(sid: str, body: Turn, request: Request, device=Depends(auth)):
        s = request.app.state.service
        session = s.session(sid, device)
        if settings.provider == 'off':
            raise HTTPException(503, 'Remote guidance is disabled. Offline detection remains available.')
        if body.requestId:
            existing = s.db.execute('SELECT id,status FROM turns WHERE session=? AND request_id=?', (sid,body.requestId)).fetchone()
            if existing:
                return dict(existing)
        if body.sceneVersion < session['scene_version']:
            raise HTTPException(409, 'This camera scene is obsolete')
        if len(dump(body.context.model_dump())) > 50000:
            raise HTTPException(422, 'Inspection context is too large')
        now = time.time()
        previous = s.db.execute('SELECT * FROM turns WHERE session=? ORDER BY created DESC LIMIT 1', (sid,)).fetchone()
        if previous and now - previous['created'] < settings.min_interval:
            raise HTTPException(429, 'Snapshot limit: wait before requesting more guidance', headers={'Retry-After':str(max(1,round(settings.min_interval-(now-previous['created']))))})
        if s.db.execute('SELECT COUNT(*) FROM turns WHERE session=?', (sid,)).fetchone()[0] >= settings.session_limit:
            raise HTTPException(429, 'This inspection reached its planning limit')
        # Account-wide limit cannot be bypassed by pairing a second device.
        if s.db.execute('SELECT COUNT(*) FROM turns WHERE created>?', (now-86400,)).fetchone()[0] >= settings.daily_limit:
            raise HTTPException(429, 'The backend reached its daily planning limit')
        tid = uid()
        context = body.context.model_dump()
        registry = s.allowed_registry(context)
        if body.context.activeTaskId and body.context.activeTaskId not in registry:
            raise HTTPException(422, 'Inspection check is unavailable for this device')
        if body.context.mode == 'research' and (not body.context.activeTaskId or registry[body.context.activeTaskId]['mode'] != 'research'):
            raise HTTPException(422, 'Expanded inspection mode requires a registered installed detector check')
        if body.context.mode == 'research' and (body.context.selectionMode != 'confirm' or not body.context.selectionConfirmed):
            raise HTTPException(422, 'Confirm the expanded check before starting its detector')
        if body.context.selectionMode == 'confirm':
            task = registry.get(body.context.activeTaskId) if body.context.activeTaskId else None
            if body.context.activeTaskId and (task is None or task['mode'] != body.context.mode or not body.context.selectionConfirmed):
                raise HTTPException(422, 'Select a supported check before starting its detector')
            if not body.context.activeTaskId and body.context.mode != 'manual':
                raise HTTPException(422, 'Exploring requires the manual camera mode')
            if any(t not in s.registry for t in body.context.dismissedTaskIds + body.context.completedTaskIds):
                raise HTTPException(422, 'Unknown dismissed or completed inspection check')
        if body.image:
            path, quality = s.snapshot(body.image, tid)
            context['backendQuality'] = quality
        elif previous and Path(previous['snapshot']).exists():
            path = previous['snapshot']
            context['backendQuality'] = json.loads(previous['context']).get('backendQuality', {})
        else:
            raise HTTPException(422, 'A fresh camera snapshot is required')
        for old in s.db.execute("SELECT id,status,scene_version FROM turns WHERE session=? AND status IN ('queued','running')", (sid,)).fetchall():
            old_context = json.loads(s.turn(sid, old['id'])['context'])
            if (old['status'] == 'queued' or old['scene_version'] < body.sceneVersion
                    or capability_key(old_context) != capability_key(context)
                    or selection_key(old_context) != selection_key(context)):
                s.cancel(old['id'])
        s.db.execute('UPDATE sessions SET scene_version=? WHERE id=?', (body.sceneVersion,sid))
        s.db.execute('INSERT INTO turns(id,session,device,request_id,scene_version,status,provider,context,snapshot,created) VALUES(?,?,?,?,?,?,?,?,?,?)',
                     (tid,sid,device,body.requestId,body.sceneVersion,'queued',settings.provider,dump(context),path,now))
        s.db.commit()
        s.signal.set()
        return {'id':tid,'status':'queued'}

    @app.get('/api/v1/sessions/{sid}/turns/{tid}')
    async def get_turn(sid: str, tid: str, request: Request, device=Depends(auth)):
        s = request.app.state.service
        s.session(sid,device)
        row = s.turn(sid,tid)
        result = {'id':tid,'status':row['status'],'sceneVersion':row['scene_version'],'provider':row['provider']}
        if row['result']:
            result['result'] = json.loads(row['result'])
        if row['error']:
            result['error'] = row['error']
        return result

    @app.post('/api/v1/sessions/{sid}/tool-results')
    async def tool_results(sid: str, body: ToolResults, request: Request, device=Depends(auth)):
        s = request.app.state.service
        session = s.session(sid,device)
        row = s.turn(sid,body.turnId)
        if body.sceneVersion != session['scene_version'] or body.sceneVersion != row['scene_version'] or row['status'] != 'completed':
            raise HTTPException(409, 'Tool results belong to an obsolete or incomplete plan')
        allowed = {a['id'] for a in json.loads(row['result'])['actions']}
        if any(x.actionId not in allowed for x in body.results):
            raise HTTPException(422, 'Unknown inspection action')
        current = s.current_context(session)
        turn_context = json.loads(row['context'])
        if capability_key(current) != capability_key(turn_context) or selection_key(current) != selection_key(turn_context):
            raise HTTPException(409, 'Tool results belong to obsolete inspection capabilities or selection')
        if (json.loads(row['context']).get('selectionMode') == 'confirm'
                or any(a.get('mode') == 'research' for a in json.loads(row['result'])['actions'])):
            actions = {a['id']:a for a in json.loads(row['result'])['actions']}
            decisions = [json.loads(r['payload']) for r in s.db.execute(
                "SELECT payload FROM events WHERE session=? AND kind='check_event' ORDER BY id",(sid,))]
            selected = None
            for event in decisions:
                if event['type'] == 'selected':
                    selected = event['taskId']
                elif event['type'] in ('stopped','completed','skipped') and event['taskId'] == selected:
                    selected = None
            if any(x.status == 'completed' and actions[x.actionId]['type'] == 'select_detector'
                   and (not selected or actions[x.actionId]['taskId'] != selected) for x in body.results):
                raise HTTPException(409, 'A detector proposal cannot execute before the user selects its check')
        s.event(sid,body.turnId,'tool_results',body.model_dump())
        return {'accepted':True}

    @app.get('/api/v1/sessions/{sid}/trace')
    async def trace(sid: str, request: Request, device=Depends(auth)):
        s = request.app.state.service
        session = s.session(sid,device,allow_closed=True)
        return {'sessionId':sid,'events':[{'type':r['kind'],'turnId':r['turn_id'],'at':r['created'],'data':json.loads(r['payload'])}
            for r in s.db.execute('SELECT * FROM events WHERE session=? ORDER BY id',(sid,)).fetchall()]}

    @app.post('/api/v1/sessions/{sid}/events')
    async def check_event(sid: str, body: CheckEvent, request: Request, device=Depends(auth)):
        s = request.app.state.service
        # Human decisions are records, not commands. A late acknowledgement after a
        # scene update or session closure must not silently disappear from the trace.
        session = s.session(sid,device,allow_closed=True)
        payload = body.model_dump()
        existing = s.db.execute('SELECT payload FROM events WHERE session=? AND client_event_id=?',(sid,body.eventId)).fetchone()
        if existing:
            if json.loads(existing['payload']) != payload:
                raise HTTPException(409, 'This event identifier was already used for a different decision')
            return {'accepted':True,'duplicate':True,'eventId':body.eventId}
        if body.taskId not in s.registry:
            raise HTTPException(422, 'Unknown inspection check')
        if not available(s.registry[body.taskId], s.current_context(session)):
            raise HTTPException(422, 'Inspection check is unavailable for this device')
        if body.sourceTurnId:
            s.turn(sid,body.sourceTurnId)
        s.db.execute('INSERT INTO events(session,turn_id,kind,payload,created,client_event_id) VALUES(?,?,?,?,?,?)',
            (sid,body.sourceTurnId,'check_event',dump(payload),time.time(),body.eventId))
        s.db.commit()
        return {'accepted':True,'duplicate':False,'eventId':body.eventId}

    @app.delete('/api/v1/sessions/{sid}')
    async def close(sid: str, request: Request, device=Depends(auth)):
        s = request.app.state.service
        s.session(sid,device)
        s.db.execute('UPDATE sessions SET closed=1 WHERE id=?',(sid,))
        for row in s.db.execute("SELECT id FROM turns WHERE session=? AND status IN ('queued','running')",(sid,)).fetchall():
            s.cancel(row['id'],'cancelled')
        s.db.commit()
        s.event(sid,None,'closed',{})
        return {'closed':True}

    return app
