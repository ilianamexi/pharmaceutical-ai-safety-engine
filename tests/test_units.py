import torch
import torch.nn.functional as F

from src.data.drug_mapping import DrugBankMapper, strip_salts
from src.data.openfda import parse_report
from src.models.focal_risk_engine import FocalLoss
from tests.synthetic import write_vocab


def test_focal_loss_reduces_to_bce():
    logits, y = torch.randn(64), (torch.rand(64) > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(FocalLoss(alpha=None, gamma=0.0)(logits, y), bce, atol=1e-6)
    assert torch.allclose(FocalLoss(alpha=0.5, gamma=0.0)(logits, y), 0.5 * bce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    y = torch.tensor([1.0])
    easy, hard = torch.tensor([4.0]), torch.tensor([-1.0])
    fl, bce = FocalLoss(alpha=None, gamma=2.0), torch.nn.BCEWithLogitsLoss()
    assert fl(easy, y) / bce(easy, y) < fl(hard, y) / bce(hard, y)


def test_focal_alpha_weights_classes():
    logits = torch.zeros(2)
    fl = FocalLoss(alpha=0.75, gamma=0.0)
    pos = fl(logits[:1], torch.ones(1))
    neg = fl(logits[1:], torch.zeros(1))
    assert torch.isclose(pos / neg, torch.tensor(3.0))


def test_parse_report_basic():
    raw = {
        "safetyreportid": "123", "receivedate": "20240105", "serious": "1", "seriousnessdeath": "1",
        "patient": {
            "patientonsetage": "6", "patientonsetageunit": "802", "patientsex": "2",
            "drug": [
                {"medicinalproduct": "LETAIRIS", "drugcharacterization": "1",
                 "activesubstance": {"activesubstancename": "AMBRISENTAN"}},
                {"medicinalproduct": "Letairis", "drugcharacterization": "1",
                 "activesubstance": {"activesubstancename": "AMBRISENTAN"}},  # duplicate -> dropped
                {"medicinalproduct": "OXYGEN", "drugcharacterization": "2"},
            ],
            "reaction": [{"reactionmeddrapt": "Cough"}, {"reactionmeddrapt": "cough"}],
        },
    }
    r = parse_report(raw)
    assert r["serious"] and r["death"] and not r["hospitalization"]
    assert r["age_years"] == 0.5 and r["sex"] == "female"
    assert [d["active_substance"] or d["name"] for d in r["drugs"]] == ["AMBRISENTAN", "OXYGEN"]
    assert r["drugs"][1]["role"] == "concomitant"
    assert r["reactions"] == ["cough"]


def test_parse_report_rejects_incomplete():
    assert parse_report({"serious": "1", "patient": {"drug": [{"medicinalproduct": "X"}]}}) is None


def test_drug_mapping(tmp_path):
    vocab = tmp_path / "vocab.csv"
    write_vocab(vocab)
    m = DrugBankMapper(vocab)
    assert strip_salts("Doxycycline Hyclate") == "DOXYCYCLINE"
    assert m.map_drug({"name": "DOXYCYCLINE HYCLATE"}) == ["DB00254"]
    assert m.map_drug({"name": "PARACETAMOL"}) == ["DB00316"]            # synonym
    assert m.map_drug({"name": "??", "unii": ["5Q7ZVV76EI"]}) == ["DB00682"]  # UNII first
    assert m.map_drug({"name": "ASPIRIN AND WARFARIN"}) == ["DB00682", "DB00945"]
    assert m.map_drug({"name": "NOT A DRUG"}) == []


def test_link_split_has_no_test_edges_in_message_passing():
    from torch_geometric.data import Data
    from torch_geometric.transforms import RandomLinkSplit
    from torch_geometric.utils import to_undirected

    torch.manual_seed(0)
    ei = to_undirected(torch.randint(0, 200, (2, 1500)), num_nodes=200)
    train, val, test = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                                       add_negative_train_samples=False)(Data(edge_index=ei, num_nodes=200))
    mp = {tuple(e) for e in train.edge_index.t().tolist()}
    test_pos = test.edge_label_index[:, test.edge_label == 1].t().tolist()
    assert not any(tuple(e) in mp or tuple(e[::-1]) in mp for e in test_pos)


def test_wikidata_vocabulary_parsing(tmp_path):
    from src.data.wikidata_vocab import build_vocabulary, normalize_drugbank_id

    assert normalize_drugbank_id("00945") == "DB00945"
    assert normalize_drugbank_id("DB00945") == "DB00945"
    assert normalize_drugbank_id("garbage") is None

    def b(**kw):
        return {k: {"value": v} for k, v in kw.items()}

    main = [b(item="Q1", db="00945", label="aspirin", unii="R16CO5Y76E"),
            b(item="Q2", db="DB00682", label="warfarin")]
    syn = [b(item="Q1", alt="acetylsalicylic acid"), b(item="Q1", alt="Aspirin")]
    rows = build_vocabulary(main, syn)
    path = tmp_path / "v.csv"
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["DrugBank ID", "Common name", "UNII", "Synonyms"])
        w.writeheader(); w.writerows(rows)
    m = DrugBankMapper(path)
    assert m.map_drug({"name": "ACETYLSALICYLIC ACID"}) == ["DB00945"]
    assert m.map_drug({"name": "?", "unii": ["R16CO5Y76E"]}) == ["DB00945"]
    assert m.map_drug({"name": "WARFARIN SODIUM"}) == ["DB00682"]


def test_degree_matched_negatives_are_valid_non_edges():
    from src.gnn.train_ddi_gnn import degree_matched_negatives
    from torch_geometric.utils import to_undirected

    torch.manual_seed(1)
    ei = to_undirected(torch.randint(0, 100, (2, 600)), num_nodes=100)
    ei = ei[:, ei[0] != ei[1]]
    pos = ei[:, :50]
    neg = degree_matched_negatives(pos, ei, ei, 100, seed=0)
    known = set(map(tuple, ei.t().tolist()))
    assert neg.shape == pos.shape
    assert all(tuple(e) not in known and e[0] != e[1] for e in neg.t().tolist())
    # every negative keeps one endpoint of its positive
    assert all(set(p) & set(n) for p, n in zip(pos.t().tolist(), neg.t().tolist()))


def test_month_windows_cover_month():
    from src.models.temporal_eval import month_windows

    assert month_windows("202502") == [("20250201", "20250207"), ("20250208", "20250214"),
                                       ("20250215", "20250221"), ("20250222", "20250228")]
    assert month_windows("202403")[-1] == ("20240322", "20240331")
