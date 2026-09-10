defmodule ExPhil.Data.FilenameTagsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.FilenameTags

  test "both sides tagged" do
    assert FilenameTags.parse("03_20_55 [RUDE] Captain Falcon + [INFP] Fox (PS).slp") ==
             [{"RUDE", "captain falcon"}, {"INFP", "fox"}]
  end

  test "one side untagged" do
    assert FilenameTags.parse("00_41_46 [SM] Falco + Fox (FD).slp") ==
             [{"SM", "falco"}, {nil, "fox"}]
  end

  test "no tags, no stage" do
    assert FilenameTags.parse("01_15_15 Falco + Captain Falcon (YI).slp") ==
             [{nil, "falco"}, {nil, "captain falcon"}]
  end

  test "hashed rank-tier names yield no entries" do
    assert FilenameTags.parse("master-master-000551fbfa951d4cce5f3f7a.slp") == []
  end

  test "Z-suffixed timestamps parse" do
    assert FilenameTags.parse("03_37_14.788Z Sheik + [=P] Fox (BF).slp") ==
             [{nil, "sheik"}, {"=P", "fox"}]
  end

  describe "subject_tag/2" do
    test "picks the subject character's tag" do
      assert FilenameTags.subject_tag("x/[RUDE] Captain Falcon + [INFP] Fox (PS).slp", "Fox") ==
               "INFP"

      assert FilenameTags.subject_tag("x/[RUDE] Captain Falcon + [INFP] Fox (PS).slp", "Captain Falcon") ==
               "RUDE"
    end

    test "nil for untagged subject side" do
      assert FilenameTags.subject_tag("x/[SM] Falco + Fox (FD).slp", "Fox") == nil
    end

    test "nil for dittos (ambiguous)" do
      assert FilenameTags.subject_tag("x/[A] Fox + [B] Fox (BF).slp", "Fox") == nil
    end

    test "nil for hashed names and absent character" do
      assert FilenameTags.subject_tag("x/master-master-00055.slp", "Fox") == nil
      assert FilenameTags.subject_tag("x/[SM] Falco + Marth (FD).slp", "Fox") == nil
    end
  end

  test "placeholder?/1" do
    assert FilenameTags.placeholder?(nil)
    assert FilenameTags.placeholder?("")
    assert FilenameTags.placeholder?("Master Player")
    refute FilenameTags.placeholder?("RUDE")
  end

  # FALCO-dir "vs" conventions (measured 2026-09-05; V2_PREP item 7)

  test "vs Game_-form: paren tag after character, bracket stage dropped" do
    assert FilenameTags.parse("x/Falco vs Link (MILO) [BF] Game_20200226T205022.slp") ==
             [{nil, "falco"}, {"MILO", "link"}]
  end

  test "vs Game_-form: untagged sides" do
    assert FilenameTags.parse("x/Falco vs Marth [FoD] Game_20200222T213755.slp") ==
             [{nil, "falco"}, {nil, "marth"}]
  end

  test "tournament long-form: prefix tag kept, costume parens denylisted" do
    path = "x/20200101 - HNC 5 - PM 1006 - Falco (Default) vs SOPH Marth (White) - Dream Land N64.slp"
    assert FilenameTags.parse(path) == [{nil, "falco"}, {"SOPH", "marth"}]
    assert FilenameTags.subject_tag(path, "Marth") == "SOPH"
    assert FilenameTags.subject_tag(path, "Falco") == nil
  end

  test "tournament long-form: both sides untagged (costumes only)" do
    path = "x/20200129 - HNC 2 - PM 0725 - Pikachu (Default) vs Falco (Default) - Dream Land N64.slp"
    assert FilenameTags.parse(path) == [{nil, "pikachu"}, {nil, "falco"}]
  end

  test "vs-form: stage abbrev in parens is not a tag (denylist, exact case)" do
    assert FilenameTags.parse("x/Falco vs Marth (BF) Game_1.slp") ==
             [{nil, "falco"}, {nil, "marth"}]

    # an ALL-CAPS player tag colliding with a color word survives
    assert FilenameTags.parse("x/Falco vs Marth (RED) Game_1.slp") ==
             [{nil, "falco"}, {"RED", "marth"}]
  end

  test "vs-form: multi-word characters tail-match longest-first" do
    path = "x/20200101 - W 1 - PM 1 - BOY Falco (Red) vs Young Link (Green) - Battlefield.slp"
    assert FilenameTags.parse(path) == [{"BOY", "falco"}, {nil, "young link"}]
  end

  test "bare hashed / Game_ names parse to []" do
    assert FilenameTags.parse("x/Game_20190309T112729.slp") == []
    assert FilenameTags.parse("x/master-master-00055.slp") == []
  end
end

defmodule ExPhil.Data.FilenameTagsDittoTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.FilenameTags

  test "ditto resolves positionally by port" do
    path = "x/[A] Fox + [B] Fox (BF).slp"
    assert FilenameTags.subject_tag(path, "Fox", 1) == "A"
    assert FilenameTags.subject_tag(path, "Fox", 2) == "B"
    # ports 3/4: position convention doesn't cover them
    assert FilenameTags.subject_tag(path, "Fox", 3) == nil
  end

  test "ditto with one untagged side" do
    path = "x/[A] Fox + Fox (BF).slp"
    assert FilenameTags.subject_tag(path, "Fox", 1) == "A"
    assert FilenameTags.subject_tag(path, "Fox", 2) == nil
  end

  test "non-ditto ignores the port (character match wins)" do
    path = "x/[RUDE] Captain Falcon + [INFP] Fox (PS).slp"
    assert FilenameTags.subject_tag(path, "Fox", 1) == "INFP"
    assert FilenameTags.subject_tag(path, "Fox", 2) == "INFP"
  end

  test "hashed names stay nil under /3" do
    assert FilenameTags.subject_tag("x/master-master-00055.slp", "Fox", 1) == nil
  end
end
