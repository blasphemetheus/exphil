defmodule ExPhil.Eval.TeacherFitTest do
  use ExUnit.Case, async: true
  alias ExPhil.Eval.TeacherFit

  test "joint likelihood includes all buttons and conditionally evaluated heads" do
    logits = %{
      buttons: List.duplicate(0.0, 8),
      main_x: [0.0, 0.0],
      main_y: [0.0, 0.0],
      c_x: [0.0, 0.0],
      c_y: [0.0, 0.0],
      shoulder: [0.0, 0.0]
    }

    target = %{buttons: List.duplicate(1, 8), main_x: 0, main_y: 0, c_x: 0, c_y: 0, shoulder: 0}
    row = TeacherFit.row(logits, target)
    assert_in_delta row.nll, 13 * :math.log(2), 1.0e-10
    assert_in_delta row.target_probability, :math.pow(2, -13), 1.0e-10
    assert row.tf_argmax_correct
    assert TeacherFit.summarize([row]).count == 1
    assert TeacherFit.summarize([]) == %{count: 0}
  end

  test "extreme confidently wrong logits stay finite and expose B and X errors" do
    logits = %{
      buttons: List.duplicate(1000.0, 8),
      main_x: [-1000.0, 1000.0],
      main_y: [-1000.0, 1000.0],
      c_x: [-1000.0, 1000.0],
      c_y: [-1000.0, 1000.0],
      shoulder: [-1000.0, 1000.0]
    }

    target = %{buttons: List.duplicate(0, 8), main_x: 0, main_y: 0, c_x: 0, c_y: 0, shoulder: 0}
    row = TeacherFit.row(logits, target)
    assert row.nll == 18000.0
    refute row.b_correct
    refute row.x_correct
    assert row.target_probability == 0.0
    assert TeacherFit.summarize([row]).tf_argmax_correct == 0.0
  end
end
