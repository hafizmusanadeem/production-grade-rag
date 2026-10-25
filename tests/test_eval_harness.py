import pytest

from evals.harness import flatten_summary, select_goldens


def test_select_is_deterministic():
    data=[{"id":str(i),"source":str(i%3)} for i in range(20)]
    assert select_goldens(data,limit=7,seed=2)==select_goldens(data,limit=7,seed=2)


def test_stratified_select_covers_sources():
    data=[{"id":str(i),"source":str(i)} for i in range(5)]
    assert {x["source"] for x in select_goldens(data,limit=5,stratify_key="source")}=={str(i) for i in range(5)}


def test_unknown_id_raises():
    with pytest.raises(ValueError,match="missing"):
        select_goldens([{"id":"a"}],ids=["missing"])


def test_flatten_summary():
    assert flatten_summary({"Contextual Recall":{"avg_score":0.5}})=={"contextual_recall.avg_score":0.5}
