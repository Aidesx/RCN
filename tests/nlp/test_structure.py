"""Tests: text structure understanding (word -> sentence -> paragraph)."""
import pytest

from docproc.nlp import analyze_structure


class TestParagraphs:
    def test_blank_line_separates_paragraphs(self):
        r = analyze_structure("First para here.\n\nSecond para here.")
        assert r["stats"]["paragraphs"] == 2
        assert [p["index"] for p in r["paragraphs"]] == [0, 1]

    def test_multiple_blank_lines_collapse(self):
        r = analyze_structure("A.\n\n\n\nB.")
        assert r["stats"]["paragraphs"] == 2

    def test_empty_and_whitespace_only(self):
        for t in ["", "   ", "\n\n  \n"]:
            r = analyze_structure(t)
            assert r["stats"]["paragraphs"] == 0
            assert r["stats"]["words"] == 0


class TestSentences:
    def test_basic_split(self):
        r = analyze_structure("One sentence here. Another follows! A question?")
        assert r["stats"]["sentences"] == 3

    def test_abbreviation_not_split(self):
        r = analyze_structure("Contact Dr. Smith today. He will reply.")
        sents = r["paragraphs"][0]["sentences"]
        assert len(sents) == 2
        assert sents[0].startswith("Contact Dr. Smith")

    def test_no_terminal_punctuation_is_one_sentence(self):
        r = analyze_structure("this has no ending")
        assert r["stats"]["sentences"] == 1

    def test_sentences_stay_inside_their_paragraph(self):
        r = analyze_structure("Para one. Has two.\n\nPara two only.")
        assert r["paragraphs"][0]["sentence_count"] == 2
        assert r["paragraphs"][1]["sentence_count"] == 1


class TestWords:
    def test_counts_and_unique(self):
        r = analyze_structure("the cat the dog. The bird flies!")
        assert r["stats"]["words"] == 7
        assert r["stats"]["unique_words"] == 5

    def test_hyphen_and_apostrophe_tokens(self):
        r = analyze_structure("state-of-the-art design isn't easy")
        words = sum(p["word_count"] for p in r["paragraphs"])
        assert words == 4  # state-of-the-art is one token

    def test_unicode_vietnamese(self):
        r = analyze_structure("Hóa đơn số 123 tổng tiền 45.000 đồng.")
        assert r["stats"]["words"] >= 6
        assert r["stats"]["sentences"] == 1


class TestContract:
    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            analyze_structure(123)

    def test_stats_consistent_with_paragraph_details(self):
        r = analyze_structure("Alpha beta gamma. Delta.\n\nEpsilon zeta.")
        total_from_paras = sum(p["word_count"] for p in r["paragraphs"])
        sent_total = sum(p["sentence_count"] for p in r["paragraphs"])
        assert r["stats"]["words"] == total_from_paras
        assert r["stats"]["sentences"] == sent_total

    def test_demo_invoice_shape(self):
        text = ("ACME - INVOICE #10482\n\nTotal due is $147.50 within 30 days.\n"
                "Please reference the number. Contact Dr. Smith.")
        r = analyze_structure(text)
        assert r["stats"]["paragraphs"] == 2
        assert r["paragraphs"][1]["sentence_count"] == 3
        assert r["stats"]["sentences"] == 4