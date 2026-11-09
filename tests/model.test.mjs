import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { predict, recommend, encodeResponses } from '../model-core.js';
const model = JSON.parse(readFileSync(new URL('../model.json', import.meta.url), 'utf8'));
const profile = {
  Country:'Canada',Age:'25-34 years old',
  Education:"Bachelor's degree (B.A., B.S., B.Eng., etc.)",Employment:'Employed',
  Role:'Developer, back-end',ICorPM:'Individual contributor',
  RoleFunctions:['Writing, editing, or generating code','Debugging or troubleshooting'],
  OrgSize:'100 to 499 employees',WorkSetting:'Remote',
  YearsCode:9,WorkExp:5,SOYears:'Between 5 and 10 years',
  LanguagesHaveWorkedWith:['Python','JavaScript'],
  CloudPlatHaveWorkedWith:['Amazon Web Services (AWS)','Docker / Docker Compose'],
  KnowMgmtHaveWorkedWith:['Notion']
};
test('exports 17 original survey inputs but 18 selected model features',()=>{
  assert.equal(model.rawFeatures.length,17);
  assert.equal(model.labels.length,15);
  assert.equal(model.intercepts.length,14);
  assert.equal(model.weights.length,14);
  const n=model.weights[0].length;
  assert.ok(n>300);
  model.weights.forEach(w=>assert.equal(w.length,n));
});
test('cumulative model always yields a normalized nonnegative ordinal distribution',()=>{
  for(const r of [{},profile,{...profile,WorkExp:null,YearsCode:null,Country:null}]) {
    const output=predict(model,r);
    assert.equal(output.probabilities.length,15);
    assert.ok(output.probabilities.every(p=>p>=0 && p<=1));
    assert.ok(Math.abs(output.probabilities.reduce((a,b)=>a+b,0)-1)<1e-11);
    assert.ok(output.medianBucket>=0 && output.medianBucket<15);
    assert.ok(output.expectedBucket>=0 && output.expectedBucket<15);
  }
});
test('engineering WorkExpSquared happens automatically and consistently',()=>{
  const v=encodeResponses(model,profile);
  const work2=model.numeric.find(n=>n.name==='WorkExpSquared');
  assert.ok(Math.abs(v.get(work2.index)-(25-work2.mean)/work2.scale)<1e-12);
});
test('recommendations add only unselected, supported tools and rank by expected bucket',()=>{
  const rec=recommend(model,profile);
  assert.ok(rec.length>10);
  assert.ok(rec.every(item=>item.support>=30));
  assert.ok(rec.every((item,i)=>i===0||rec[i-1].change>=item.change-1e-12));
  assert.ok(!rec.some(item=>item.name==='Python' && item.feature==='LanguagesHaveWorkedWith'));
  assert.ok(!rec.some(item=>item.name.includes('None of the above')));
});
