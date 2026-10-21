import {readFileSync} from 'node:fs';import {unzipSync} from 'fflate';
const filename=process.argv[2]??'delivery/'+JSON.parse(readFileSync('delivery/build-metadata.json','utf8')).apk;
const libraries=unzipSync(new Uint8Array(readFileSync(filename)),{filter:f=>/^lib\/arm64-v8a\/.*\.so$/.test(f.name)});
let count=0;
for(const [name,data]of Object.entries(libraries)){
  const b=Buffer.from(data);if(b[0]!==127||b.subarray(1,4).toString()!=='ELF'||b[4]!==2||b[5]!==1)throw Error('Unexpected ELF format: '+name);
  const offset=Number(b.readBigUInt64LE(32)),size=b.readUInt16LE(54),entries=b.readUInt16LE(56);
  for(let i=0;i<entries;i++){const p=offset+i*size;if(b.readUInt32LE(p)!==1)continue;
    const alignment=Number(b.readBigUInt64LE(p+48));if(alignment<16384)throw Error(`Non-16KB ELF LOAD segment: ${name}, align=${alignment}`);
  }
  count++;console.log('16KB ELF alignment: '+name);
}
if(count<2)throw Error('Native inference libraries missing');
console.log(`${count} ARM64 native libraries passed ELF alignment inspection.`);
