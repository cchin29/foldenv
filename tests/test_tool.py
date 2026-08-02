"""M8 (interface-agnostic) tool-wrapper tests. Schema/validation are offline; the
result-shape tests are live (network + mkdssp)."""
import json
import pathlib
import shutil

import pytest

from foldenv import config, context, tool

pytest.importorskip("Bio")


# --- offline: spec shapes + argument validation ---------------------------------------

def test_spec_styles():
    a = tool.tool_spec("anthropic")
    assert a["name"] == "get_structural_context" and "input_schema" in a
    o = tool.tool_spec("openai")
    assert o["type"] == "function" and o["function"]["parameters"] == tool.INPUT_SCHEMA
    p = tool.tool_spec("plain")
    assert "input_schema" in p and "output_schema" in p
    with pytest.raises(ValueError):
        tool.tool_spec("nope")


def test_committed_spec_in_sync():
    # the committed tool_spec.json must match what the code produces (drift guard)
    path = pathlib.Path(tool.__file__).with_name("tool_spec.json")
    assert path.is_file()
    assert json.loads(path.read_text()) == tool.tool_spec("plain")


def test_invoke_validation_offline():
    with pytest.raises(ValueError):                       # missing required args
        tool.invoke({"uniprot_id": "P62593"})
    with pytest.raises(TypeError):                        # position not int
        tool.invoke({"uniprot_id": "P62593", "position": "150"})
    with pytest.raises(TypeError):                        # bool is not a valid position
        tool.invoke({"uniprot_id": "P62593", "position": True})
    with pytest.raises(TypeError):                        # uniprot_id not str
        tool.invoke({"uniprot_id": 5, "position": 1})
    with pytest.raises(ValueError):                       # position out of 1-based range
        tool.invoke({"uniprot_id": "P62593", "position": 0})
    with pytest.raises(ValueError):                       # malformed accession fails fast
        tool.invoke({"uniprot_id": "not an accession!", "position": 1})
    with pytest.raises(TypeError):                        # stringy bool must not slip through
        tool.invoke({"uniprot_id": "P62593", "position": 1, "include_embedding": "false"})


def test_input_schema_required_fields():
    assert set(tool.INPUT_SCHEMA["required"]) == {"uniprot_id", "position"}
    assert tool.INPUT_SCHEMA["additionalProperties"] is False


# --- live: result shape (embedding off by default) ------------------------------------

def _online():
    import requests
    try:
        requests.get(config.load()["alphafold"]["api_base"] + "/P62593", timeout=15)
        return True
    except Exception:
        return False


live = pytest.mark.skipif(
    not _online() or shutil.which("mkdssp") is None, reason="needs AlphaFold-DB + mkdssp"
)


@pytest.fixture(autouse=True)
def _clear():
    context.clear_cache()
    yield
    context.clear_cache()


@live
def test_invoke_default_omits_embedding():
    out = tool.invoke({"uniprot_id": "P62593", "position": 150})
    # structural fields present; embedding omitted (null) so no PLM forward pass
    assert out["secondary_structure"] in ("H", "E", "C")
    assert out["embedding"] is None and out["embedding_model"] is None
    assert isinstance(out["contact_count"], int)
    json.dumps(out, allow_nan=False)   # strict JSON


@live
def test_invoke_accepts_json_string():
    out = tool.invoke('{"uniprot_id": "P62593", "position": 1}')
    assert out["wildtype_aa"] == "M" and out["position"] == 1


# --- heavy: include_embedding=true actually returns the vector ------------------------

@pytest.mark.skipif(__import__("os").environ.get("RUN_HEAVY_EMB") != "1",
                    reason="set RUN_HEAVY_EMB=1")
@live
def test_invoke_with_embedding():
    out = tool.invoke({"uniprot_id": "P62593", "position": 150, "include_embedding": True})
    assert out["embedding_model"] == "ankh"
    assert isinstance(out["embedding"], list) and len(out["embedding"]) == 1536
