import { predict, recommend } from './model-core.js';

const GROUPS = [
  { number: '01', title: 'Background', fields: ['Country','Age','Education','Employment','EmploymentAddl'] },
  { number: '02', title: 'Your work', fields: ['Role','RoleFunctions','ICorPM','OrgSize','WorkSetting','YearsCode','WorkExp'] },
  { number: '03', title: 'Finding information', fields: ['SOYears','KnowledgeQ'] },
  { number: '04', title: 'Technologies & tools', fields: ['LanguagesHaveWorkedWith','CloudPlatHaveWorkedWith','KnowMgmtHaveWorkedWith'] },
];
const META = {
  Country:['Country of residence','Where do you currently live?'],
  Age:['Age range','Select your age group from the original survey.'],
  Education:['Highest education completed','Select the closest survey response.'],
  Employment:['Current employment status','Your primary employment status.'],
  EmploymentAddl:['Additional responsibilities or work','Select all that apply, including “None of the above” if appropriate.'],
  Role:['Primary professional role','Choose the role that most closely reflects your work.'],
  RoleFunctions:['Activities you perform at work','Select all functions you regularly perform.'],
  ICorPM:['Individual contributor or manager','How would you describe your responsibilities?'],
  OrgSize:['Organization size','Approximate number of employees.'],
  WorkSetting:['Work arrangement','Select the closest working arrangement.'],
  YearsCode:['Years you have been coding','Number of years since you first started coding.'],
  WorkExp:['Years of professional experience','Professional work experience in years; the model derives experience squared automatically.'],
  SOYears:['Years using Stack Overflow','Select the original survey response.'],
  KnowledgeQ:['Where you find technical answers','Select all sources you use to find information.'],
  LanguagesHaveWorkedWith:['Programming languages used','Select languages you have worked with.'],
  CloudPlatHaveWorkedWith:['Platforms, cloud services and tools used','The survey groups cloud services, developer tools and platforms together.'],
  KnowMgmtHaveWorkedWith:['Knowledge management tools used','Select platforms you have used professionally.'],
};
const SHORT_LABELS = [
  '$0–<10k','$10–<20k','$20–<30k','$30–<40k','$40–<50k',
  '$50–<75k','$75–<100k','$100–<125k','$125–<150k',
  '$150–<200k','$200–<250k','$250–<300k','$300–<400k',
  '$400–≤500k','>$500k'
];
const DEMO = {
  Country:'Canada',Age:'25-34 years old',
  Education:"Bachelor's degree (B.A., B.S., B.Eng., etc.)", Employment:'Employed',
  EmploymentAddl:['None of the above'], Role:'Developer, back-end',
  RoleFunctions:['Writing, editing, or generating code','Debugging or troubleshooting','Writing or maintaining tests','Reviewing code, pull requests, or technical changes'],
  ICorPM:'Individual contributor', OrgSize:'100 to 499 employees', WorkSetting:'Remote',
  YearsCode:9, WorkExp:5, SOYears:'Between 5 and 10 years',
  KnowledgeQ:['Code repositories or code comments','Official docs','Stack Overflow'],
  LanguagesHaveWorkedWith:['JavaScript','Python','SQL'],
  CloudPlatHaveWorkedWith:['Docker / Docker Compose','Amazon Web Services (AWS)'],
  KnowMgmtHaveWorkedWith:['Notion'],
};

let model;
let responses = {};
let hasCalculated = false;
let moreShown = false;
let currentRecommendations = [];
let toastTimer;
const $ = id => document.getElementById(id);

function element(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function fieldSpec(name) {
  const numerical = model.numeric.find(f => f.name === name);
  if (numerical) return { kind:'number', data:numerical };
  const single = model.single.find(f => f.name === name);
  if (single) return { kind:'single', data:single };
  const multi = model.multi.find(f => f.name === name);
  if (multi) return { kind:'multi', data:multi };
  throw new Error(`Unrecognized field ${name}`);
}

function answered(name) {
  const answer = responses[name];
  return Array.isArray(answer) ? answer.length > 0 : answer !== undefined && answer !== null && answer !== '';
}
function updateProgress() {
  const total = model.rawFeatures.length;
  const count = model.rawFeatures.filter(answered).length;
  const percentage = Math.round(count / total * 100);
  $('progress-label').textContent = `${count} of ${total} answered`;
  $('progress-percent').textContent = `${percentage}%`;
  $('progress-fill').style.width = `${percentage}%`;
}
function formChanged() {
  updateProgress();
  if (hasCalculated) updateAnalysis();
}
function buildQuestion(name) {
  const spec = fieldSpec(name);
  const div = element('div','question');
  const label = element('label','question-label', META[name]?.[0] || name);
  label.htmlFor = `input-${name}`;
  div.appendChild(label);
  div.appendChild(element('p','question-help',META[name]?.[1] || ''));

  if (spec.kind === 'number') {
    const input = element('input');
    input.type = 'number'; input.min = '0'; input.max = '100'; input.step = '1';
    input.id = `input-${name}`; input.name = name; input.placeholder = 'Leave blank if unknown';
    input.value = responses[name] === null || responses[name] === undefined ? '' : responses[name];
    input.addEventListener('input', () => {
      const value = input.value;
      responses[name] = value === '' ? null : Math.max(0, Math.min(100, Number(value)));
      formChanged();
    });
    input.addEventListener('change', () => {
      if (input.value !== '' && (Number(input.value)<0 || Number(input.value)>100)) {
        input.value = String(responses[name]);
        toast('Please enter years between 0 and 100.');
      }
    });
    div.appendChild(input);
  } else if (spec.kind === 'single') {
    const select = element('select'); select.id=`input-${name}`; select.name=name;
    const blank=element('option'); blank.value=''; blank.textContent='Select an answer (or leave blank)'; select.appendChild(blank);
    for (const option of spec.data.options) {
      if (option === '__MISSING__') continue;
      const node=element('option'); node.value=option; node.textContent=option;
      select.appendChild(node);
    }
    select.value = responses[name] ?? '';
    select.addEventListener('change',()=>{responses[name]=select.value || null;formChanged()});
    div.appendChild(select);
  } else {
    const details = element('details','multi-picker');
    details.id=`input-${name}`;
    const summary = element('summary');
    const summaryValue=element('span','selected-summary is-empty','Select any that apply');
    summary.appendChild(summaryValue);
    summary.appendChild(element('span','dropdown-icon','⌄'));
    details.appendChild(summary);

    const panel=element('div','multi-options');
    const search=element('input'); search.type='search';search.placeholder='Search options…';
    search.setAttribute('aria-label', `Search ${META[name]?.[0] || name} choices`);
    panel.appendChild(search);
    const scroll=element('div','multi-scroll');
    const opts=[...spec.data.options].filter(x=>x.value!=='__MISSING__')
       .sort((a,b)=> b.support-a.support || a.value.localeCompare(b.value));
    const allRows=[];
    for (const option of opts) {
      const optionLabel=element('label','multi-choice');
      const checkbox=element('input'); checkbox.type='checkbox';checkbox.value=option.value;
      checkbox.checked=(responses[name] || []).includes(option.value);
      const text=element('span','',option.value);
      optionLabel.appendChild(checkbox);optionLabel.appendChild(text);
      optionLabel.appendChild(element('span','multi-count',option.support.toLocaleString('en-US')));
      checkbox.addEventListener('change', () => {
        const set=new Set(responses[name] || []);
        checkbox.checked ? set.add(option.value) : set.delete(option.value);
        responses[name]=[...set];
        updateSummary();formChanged();
      });
      allRows.push({row:optionLabel,value:option.value});
      scroll.appendChild(optionLabel);
    }
    const noResults=element('div','multi-no-results hidden','No matching options');
    scroll.appendChild(noResults); panel.appendChild(scroll);
    panel.appendChild(element('div','multi-footer','The counts indicate how often each answer appeared in training.'));
    details.appendChild(panel);

    function updateSummary() {
      const selected=responses[name] || [];
      summaryValue.textContent = selected.length === 0 ? 'Select any that apply'
        : selected.length===1 ? selected[0] : `${selected.length} selected · ${selected.slice(0,2).join(', ')}${selected.length>2 ? '…' : ''}`;
      summaryValue.classList.toggle('is-empty',!selected.length);
    }
    updateSummary();
    search.addEventListener('input',()=>{
      const query=search.value.trim().toLocaleLowerCase();
      let visible=0;
      for (const {row,value} of allRows) {
        const show=value.toLocaleLowerCase().includes(query);
        row.classList.toggle('hidden',!show);
        if (show) visible++;
      }
      noResults.classList.toggle('hidden',visible>0);
    });
    div.appendChild(details);
  }
  return div;
}

function renderQuestions() {
  const root=$('questions');root.replaceChildren();
  for (const group of GROUPS) {
    const details=element('details','form-group');details.open=true;
    const summary=element('summary');
    const title=element('span','group-title');
    title.appendChild(element('span','group-no',group.number));
    title.appendChild(element('strong','',group.title));
    summary.appendChild(title);
    summary.appendChild(element('span','group-chevron','⌄'));
    details.appendChild(summary);
    const body=element('div','group-body');
    for (const name of group.fields) body.appendChild(buildQuestion(name));
    details.appendChild(body);root.appendChild(details);
  }
  updateProgress();
}
function fmt(n,decimals=1){return (n*100).toFixed(decimals)+'%'}
function salaryLabel(index) { return model.labels[index] || '—'; }
function updateAnalysis() {
  try {
    const result=predict(model,responses);
    currentRecommendations=recommend(model,responses,result);
    renderResults(result);
    renderRecommendations(currentRecommendations);
    $('empty-state').classList.add('hidden');
    $('prediction-content').classList.remove('hidden');
    $('recommendation-empty').classList.add('hidden');
    $('recommendation-content').classList.remove('hidden');
    $('status').textContent = 'Model calculated';
    $('status').classList.add('ready');
  } catch (error) {
    console.error(error);
    toast('The model could not calculate a prediction. Please check the answers.');
  }
}
function renderResults(result) {
  $('median-text').textContent=salaryLabel(result.medianBucket);
  $('median-sub').textContent=`Bucket ${result.medianBucket} of 14 · expected position ${result.expectedBucket.toFixed(2)}`;
  $('mode-text').textContent=salaryLabel(result.modalBucket);
  $('mode-sub').textContent=`${fmt(result.probabilities[result.modalBucket])} probability`;
  $('higher-text').textContent=fmt(result.above100k);
  $('range-text').textContent=`${salaryLabel(result.interval80[0])} to ${salaryLabel(result.interval80[1])}`;
  const graph=$('probability-chart');graph.replaceChildren();
  const maxProb=Math.max(...result.probabilities);
  result.probabilities.forEach((p,i)=>{
    const row=element('div',`prob-row${i===result.medianBucket?' median':''}`);
    const label=element('span','prob-label',SHORT_LABELS[i]);label.title=salaryLabel(i);
    const track=element('div','prob-track');
    const bar=element('div','prob-bar');
    bar.style.width=`${(maxProb > 0 ? (p/maxProb*100) : 0).toFixed(3)}%`;
    track.title=`${salaryLabel(i)}: ${fmt(p,2)} probability`;
    track.appendChild(bar);
    row.appendChild(label);row.appendChild(track);
    row.appendChild(element('span','prob-value',fmt(p)));
    graph.appendChild(row);
  });
  graph.setAttribute('aria-label',`Probabilities over 15 annual salary ranges. Median prediction ${salaryLabel(result.medianBucket)}. Most probable range ${salaryLabel(result.modalBucket)} at ${fmt(result.probabilities[result.modalBucket])}.`);
}

function displayName(feature) {
  return ({LanguagesHaveWorkedWith:'Programming languages',CloudPlatHaveWorkedWith:'Platforms and tools',KnowMgmtHaveWorkedWith:'Knowledge management'})[feature] || feature;
}
function recRow(rec,rank) {
  const card=element('article','rec-card');
  card.appendChild(element('span','rec-rank',String(rank)));
  const info=element('div');
  info.appendChild(element('p','rec-name',rec.name));
  info.appendChild(element('span','rec-category',`${displayName(rec.feature)} · ${rec.support.toLocaleString('en-US')} training respondents`));
  card.appendChild(info);
  const value=element('div','rec-change');
  const sign=rec.change>=0?'+':'';
  value.appendChild(element('strong',rec.change<0?'negative':'',`${sign}${rec.change.toFixed(3)} buckets`));
  value.appendChild(element('small','',`${rec.changeHigh>=0?'+':''}${(rec.changeHigh*100).toFixed(2)} pp at ≥$100k`));
  card.appendChild(value);
  const tryButton=element('button','rec-try','Add to my profile ↗');
  tryButton.type='button';tryButton.setAttribute('aria-label',`Add ${rec.name} and recalculate`);
  tryButton.addEventListener('click',()=>{
    const already=new Set(responses[rec.feature] || []);
    already.add(rec.name);
    responses[rec.feature]=[...already].sort();
    renderQuestions(); updateAnalysis();
    toast(`Added ${rec.name}. Prediction and ranking updated.`);
  });
  card.appendChild(tryButton);
  return card;
}
function renderRecommendations(ranking) {
  const root=$('recommendation-list');root.replaceChildren();
  const positive=ranking.filter(v=>v.change>1e-9);
  if (!positive.length) {
    root.appendChild(element('p','recommendation-placeholder', 'No positive one-tool additions were found among the supported, unselected choices for this profile.'));
    $('show-all-button').classList.add('hidden');
    return;
  }
  const show=positive.slice(0,moreShown?10:3);
  show.forEach((rec,index)=>root.appendChild(recRow(rec,index+1)));
  const showMore=positive.length>3;
  $('show-all-button').classList.toggle('hidden',!showMore);
  $('show-all-button').setAttribute('aria-expanded',String(moreShown));
  $('show-all-button').textContent=moreShown?'Show top 3 only ↑':`Show more ranked options (${Math.min(positive.length,10)}) ↓`;
}

function toast(message){
  const node=$('toast');node.textContent=message;node.classList.remove('hidden');
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>node.classList.add('hidden'),3600);
}
function setDemo() {
  const valid={};
  for (const [name,value] of Object.entries(DEMO)) {
    const spec=fieldSpec(name);
    if (spec.kind === 'single') valid[name]=spec.data.options.includes(value)?value:null;
    else if (spec.kind === 'multi') valid[name]=value.filter(v=>spec.data.options.some(o=>o.value===v));
    else valid[name]=value;
  }
  responses=valid;
  moreShown=false;
  renderQuestions();
  // Sample profile demonstrates the site immediately, without using a private respondent.
  hasCalculated=true;updateAnalysis();
  toast('Sample profile loaded. You can edit any answer.');
}
function clearAnswers() {
  responses={};hasCalculated=false;moreShown=false;
  renderQuestions();
  $('empty-state').classList.remove('hidden');
  $('prediction-content').classList.add('hidden');
  $('recommendation-empty').classList.remove('hidden');
  $('recommendation-content').classList.add('hidden');
  $('status').textContent='Awaiting profile';$('status').classList.remove('ready');
  toast('All answers cleared.');
}

async function init() {
  try {
    const response=await fetch('./model.json', { cache:'no-cache' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    model=await response.json();
    if (model.labels?.length !== 15 || model.intercepts?.length !== 14 || model.rawFeatures?.length !== 17)
      throw new Error('Unexpected model configuration');
    renderQuestions();
    $('predict-button').disabled=false;
    $('sample-button').disabled=false;
    $('clear-button').disabled=false;
    $('profile-form').addEventListener('submit',event=>{
      event.preventDefault();hasCalculated=true;moreShown=false;updateAnalysis();
      if (matchMedia('(max-width:760px)').matches) $('results').scrollIntoView({behavior:'smooth',block:'start'});
      if (model.rawFeatures.filter(answered).length < 5)
        toast('Most fields are unanswered, so this prediction depends heavily on imputed values.');
    });
    $('sample-button').addEventListener('click',setDemo);
    $('clear-button').addEventListener('click',clearAnswers);
    $('show-all-button').addEventListener('click',()=>{
      moreShown=!moreShown;renderRecommendations(currentRecommendations);
    });
  } catch(error) {
    console.error('Unable to load model:',error);
    $('questions').replaceChildren(element('div','error-banner',
      'Could not load model.json. Publish the entire website folder to GitHub Pages, or preview it using a local HTTP server rather than opening index.html as a file.'));
    toast('Model unavailable. See the message in the questionnaire.');
  }
}
init();
