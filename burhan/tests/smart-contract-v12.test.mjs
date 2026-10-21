import test from 'node:test';
import assert from 'node:assert/strict';
import {validatePlan} from '../src/lib/smart-contract.mjs';

const plan={scene:{environment:'indoor',description:'Wall',confidence:.8},guidance:'Hold still',reason:'Check the joint',suggestions:[],actions:[],needsHumanReview:true};
test('guidance target survives provider normalization and older plans remain readable',()=>{
 assert.equal(validatePlan({...plan,guidanceTaskId:'visual-cracks'}).guidanceTaskId,'visual-cracks');
 assert.equal(validatePlan(plan).guidanceTaskId,null);
 assert.equal(validatePlan({...plan,guidanceTaskId:{untrusted:true}}).guidanceTaskId,null);
 assert.equal(validatePlan({...plan,guidanceTaskId:'x'.repeat(200)}).guidanceTaskId.length,160);
});
