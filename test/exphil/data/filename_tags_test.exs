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
end
