"""
Graph-RAG 원인 역추적 (기획서 F-04)

공정 지식그래프(Process KG)에서 예측 패턴 → 유발 공정 → 원인 → 설비 → 근거를
검색해 '설명 가능한' 리포트를 생성. LLM 없이도 구조적 리포트로 동작하며,
USE_LLM=True + transformers 설치 시 로컬 LLM(Qwen2.5)으로 문장화.
"""
from __future__ import annotations
import networkx as nx

TRIPLES = [
    ('Center','causes_process','CMP_Polishing'), ('CMP_Polishing','via_cause','Uneven_polish_pressure_slurry'), ('CMP_Polishing','on_equip','CMP_tool'),
    ('Donut','causes_process','Deposition_Litho'), ('Deposition_Litho','via_cause','Coating_thickness_variation'), ('Deposition_Litho','on_equip','Coater_Scanner'),
    ('Edge_Loc','causes_process','Etch'), ('Etch','via_cause','Edge_plasma_nonuniformity'), ('Etch','on_equip','Plasma_etcher'),
    ('Edge_Ring','causes_process','Anneal_Etch'), ('Anneal_Etch','via_cause','Edge_temperature_etch_nonuniformity'), ('Anneal_Etch','on_equip','Anneal_furnace'),
    ('Loc','causes_process','Lithography'), ('Lithography','via_cause','Focus_deviation'), ('Lithography','on_equip','Scanner_Stepper'),
    ('Near_Full','causes_process','Wide_process_fault'), ('Wide_process_fault','via_cause','Equipment_recipe_major_fault'), ('Wide_process_fault','on_equip','Fab_line'),
    ('Scratch','causes_process','Handling_Dicing'), ('Handling_Dicing','via_cause','Physical_scratch_handling'), ('Handling_Dicing','on_equip','Wafer_robot_arm'),
    ('Random','causes_process','Cleaning'), ('Cleaning','via_cause','Cleanroom_particle_contamination'), ('Cleaning','on_equip','Cleaning_bath'),
    ('Center','evidence','GlobalSino_pattern_cause'), ('Edge_Ring','evidence','Wavelet2023_frequency'),
    ('Scratch','evidence','GlobalSino_pattern_cause'), ('Center','evidence','Wang2020_MixedWM38'),
]

HUMAN = {
    'CMP_Polishing':'CMP(화학기계연마)','Deposition_Litho':'증착/노광','Etch':'식각','Anneal_Etch':'어닐링/식각',
    'Lithography':'노광','Wide_process_fault':'광범위 공정 이상','Handling_Dicing':'이송/다이싱','Cleaning':'세정',
    'Uneven_polish_pressure_slurry':'연마 압력/슬러리 불균일','Coating_thickness_variation':'도포 두께 편차',
    'Edge_plasma_nonuniformity':'가장자리 플라즈마 분포 불균일','Edge_temperature_etch_nonuniformity':'가장자리 온도/식각 불균일',
    'Focus_deviation':'초점 이탈','Equipment_recipe_major_fault':'설비/레시피 대규모 이상',
    'Physical_scratch_handling':'핸들링 중 물리적 긁힘','Cleanroom_particle_contamination':'클린룸 입자 오염',
}
# 공정 단계 번호(리포트 문구용)
PROC_STEP = {
    'Lithography':'1단계: 노광', 'Etch':'2단계: 식각', 'CMP_Polishing':'3단계: 화학 기계적 연마',
    'Deposition_Litho':'증착/노광 공정', 'Anneal_Etch':'어닐링/식각 공정',
    'Wide_process_fault':'광범위 공정', 'Handling_Dicing':'이송/다이싱 공정', 'Cleaning':'세정 공정',
}
# F-05 드롭다운용 공정 목록
PROCESS_CHOICES = ['자동 인식','노광(Lithography)','식각(Etch)','화학기계연마(CMP)','증착(Deposition)',
                   '어닐링(Anneal)','세정(Cleaning)','이송/다이싱(Handling/Dicing)']

def _h(x): return HUMAN.get(x, x)

def build_graph():
    G = nx.MultiDiGraph()
    for s, r, o in TRIPLES:
        G.add_edge(s, o, rel=r)
    return G

G = build_graph()

def retrieve(patterns):
    facts = []
    for p in patterns:
        for _, o, d in G.out_edges(p, data=True):
            if d['rel'] == 'causes_process':
                proc = o
                facts.append((p, '유발 공정', _h(proc)))
                for _, o2, d2 in G.out_edges(proc, data=True):
                    if d2['rel'] == 'via_cause': facts.append((_h(proc), '원인', _h(o2)))
                    if d2['rel'] == 'on_equip': facts.append((_h(proc), '설비', o2))
            if d['rel'] == 'evidence':
                facts.append((p, '근거', o))
    return facts

def primary_process(patterns):
    """가장 대표적인 유발 공정 문구(신뢰도 표기용)."""
    for p in patterns:
        for _, o, d in G.out_edges(p, data=True):
            if d['rel'] == 'causes_process':
                return PROC_STEP.get(o, _h(o))
    return '미상 공정'

def report(patterns, confidence=None, use_llm=False):
    """설명 가능한 원인 역추적 리포트(문자열)."""
    if not patterns:
        return '결함 패턴이 탐지되지 않았습니다. (정상 웨이퍼로 판단)'
    if use_llm:
        try:
            out = _llm_report(patterns)
            if out:
                return out
        except Exception:
            pass
    lines = []
    proc = primary_process(patterns)
    conf = f' (신뢰도 {confidence:.0%})' if confidence is not None else ''
    lines.append(f'현재 이미지는 [{proc}]에서 발생한 불량으로 판단됨{conf}.')
    lines.append('')
    for p in patterns:
        procs = [o for _, o, d in G.out_edges(p, data=True) if d['rel'] == 'causes_process']
        evs = [o for _, o, d in G.out_edges(p, data=True) if d['rel'] == 'evidence']
        for pr in procs:
            causes = [_h(o2) for _, o2, d2 in G.out_edges(pr, data=True) if d2['rel'] == 'via_cause']
            equips = [o2 for _, o2, d2 in G.out_edges(pr, data=True) if d2['rel'] == 'on_equip']
            cite = ''.join(f'[{e}]' for e in evs)
            lines.append(f'· {p}: {_h(pr)} 공정의 {", ".join(causes)}(으)로 판단. '
                         f'관련 설비: {", ".join(equips)} {cite}')
    return '\n'.join(lines)

def _llm_report(patterns):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        return None
    global _TOK, _LM
    try:
        _TOK
    except NameError:
        name = 'Qwen/Qwen2.5-3B-Instruct'
        _TOK = AutoTokenizer.from_pretrained(name)
        _LM = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map='auto')
    ctx = '\n'.join('- %s | %s | %s' % f for f in retrieve(patterns))
    prompt = ('너는 반도체 공정 엔지니어다. 아래 지식그래프 사실만 사용해 3~5문장 원인 리포트를 쓰고, '
              '근거는 [키] 형태로 인용하라. 사실에 없는 내용은 지어내지 마라.\n'
              f'탐지 패턴: {", ".join(patterns)}\n지식그래프 사실:\n{ctx}')
    msgs = [{'role': 'user', 'content': prompt}]
    text = _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = _TOK(text, return_tensors='pt').to(_LM.device)
    out = _LM.generate(**inp, max_new_tokens=300, do_sample=False)
    return _TOK.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
