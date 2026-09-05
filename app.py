"""Streamlit interface for CycAMP-FusionDemo."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.predict import predict
from cycamp.structure import CANDIDATE_CONFORMER_DISCLAIMER


def _parse_bonds(text: str) -> list[tuple[int, int]]:
    pairs = []
    normalized = text.replace("，", ",").replace("－", "-").replace("—", "-")
    for item in (part.strip() for part in normalized.split(",") if part.strip()):
        pieces = item.replace("-", ":").split(":")
        if len(pieces) != 2:
            raise ValueError("键连接格式应为 1-6,2-5")
        try:
            pairs.append((int(pieces[0]), int(pieces[1])))
        except ValueError as exc:
            raise ValueError("键连接必须使用整数残基编号，例如 1-6,2-5") from exc
    return pairs


def _score_text(score: float | None) -> str:
    """Format a model score without implying that unavailable means zero."""

    return "不可用" if score is None else f"{score:.3f}"


def _show_warning(st, message: str) -> None:
    """Render expected missing-artifact notices less severely than failures."""

    if "缺失" in message or "未加载" in message or "只支持" in message:
        st.info(message)
    else:
        st.warning(message)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="CycAMP Fusion Demo", page_icon="🧬", layout="wide")
    st.title("抗菌环肽活性预测 Demo")
    st.caption("M0人工特征 · M1冻结ESM-2 · M1-LoRA · M2序列—三维融合")
    st.info("本工具输出模型预测得分和实验验证优先级建议，不判定候选肽是否具有真实抗菌活性。")
    sequence = st.text_area("环肽序列", value="KLVFF", help="输入5–50位标准氨基酸单字母代码")
    cyclization = st.selectbox(
        "成环类型", ("head_to_tail", "disulfide", "sidechain", "other"),
        format_func=lambda value: {"head_to_tail": "头尾成环", "disulfide": "二硫键成环",
                                   "sidechain": "侧链成环", "other": "其他"}[value],
    )
    bonds_text = st.text_input("可选键连接（1开始编号）", placeholder="例如 1-6,2-5")
    if st.button("开始预测", type="primary"):
        try:
            result = predict(
                sequence, cyclization, _parse_bonds(bonds_text),
                model_dir=PROJECT_ROOT / "artifacts" / "models",
                structure_output_dir=PROJECT_ROOT / "artifacts" / "structures" / "demo",
            )
        except ValueError as exc:
            st.error(f"输入无效：{exc}")
            return
        except Exception as exc:
            st.error(f"预测流程运行失败：{type(exc).__name__}: {exc}")
            return
        st.subheader(f"实验验证优先级建议：{result.recommendation}")
        st.caption("该建议只根据当前可用模型产生；‘不可用’不是低分，也不会按0分参与规则。")
        columns = st.columns(4)
        for column, label, score in zip(
            columns, ("M0", "M1", "M1-LoRA", "M2"),
            (result.m0_score, result.m1_score, result.m1_lora_score, result.m2_score),
        ):
            column.metric(f"{label} 模型预测得分", _score_text(score))
        st.write("规范化序列：", result.normalized_sequence)
        st.write("规范环形序列：", result.canonical_sequence)
        st.subheader("基础理化描述符")
        st.dataframe(
            [{"性质": key, "数值": round(value, 4)} for key, value in result.basic_descriptors.items()],
            use_container_width=True, hide_index=True,
        )
        if result.structure_file and Path(result.structure_file).exists():
            st.subheader("三维候选构象")
            try:
                import py3Dmol
                import streamlit.components.v1 as components

                sdf = Path(result.structure_file).read_text(encoding="utf-8", errors="replace")
                view = py3Dmol.view(width=700, height=420)
                view.addModel(sdf, "sdf")
                view.setStyle({"stick": {}})
                view.zoomTo()
                components.html(view._make_html(), height=440)
            except Exception as exc:
                st.info(f"候选构象已保存，但当前无法嵌入显示：{exc}")
        else:
            st.info("本次未生成可显示的三维候选构象；其他可用模型得分仍可独立查看。")
        for warning in result.warnings:
            _show_warning(st, warning)
    st.divider()
    st.caption("模型预测不能替代体外抗菌实验。" + CANDIDATE_CONFORMER_DISCLAIMER)


if __name__ == "__main__":
    main()
