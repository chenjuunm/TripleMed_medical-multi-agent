"""Offline tests: no embedding downloads or writes to the real vector DB."""
import unittest
from unittest.mock import Mock

from langchain_community.retrievers import BM25Retriever
from rag_system import mock_guidelines, _add_missing_demo_documents, _tokenize_for_bm25


class DemoGuidelineTests(unittest.TestCase):
    def test_corpus_has_unique_traceable_demo_sources(self):
        self.assertEqual(len(mock_guidelines), 15)
        ids = [document.metadata["source_id"] for document in mock_guidelines]
        self.assertEqual(len(ids), len(set(ids)))
        for document in mock_guidelines:
            self.assertIs(document.metadata["is_demo"], True)
            self.assertTrue(document.metadata["source"])
            self.assertTrue(document.metadata["version"])
        for document in mock_guidelines[3:]:
            self.assertIn("仅用于软件检索测试", document.page_content)
            self.assertEqual(document.metadata["data_kind"], "synthetic_retrieval_fixture")

    def test_old_uuid_collection_only_gets_missing_documents(self):
        store = Mock()
        store.get.return_value = {"metadatas": [doc.metadata for doc in mock_guidelines[:3]]}
        self.assertEqual(_add_missing_demo_documents(store), 12)
        store.get.assert_called_once_with(where={"is_demo": True}, include=["metadatas"])
        call = store.add_documents.call_args.kwargs
        self.assertEqual(call["documents"], mock_guidelines[3:])
        self.assertEqual(call["ids"], [f"bundled-demo:{doc.metadata['source_id']}" for doc in mock_guidelines[3:]])
        store.delete.assert_not_called()

    def test_complete_collection_does_not_duplicate_documents(self):
        store = Mock()
        store.get.return_value = {"metadatas": [doc.metadata for doc in mock_guidelines]}
        self.assertEqual(_add_missing_demo_documents(store), 0)
        store.add_documents.assert_not_called()

    def test_empty_collection_gets_all_demos(self):
        store = Mock()
        store.get.return_value = {"metadatas": []}
        self.assertEqual(_add_missing_demo_documents(store), 15)

    def test_unrelated_metadata_does_not_prevent_additions(self):
        store = Mock()
        store.get.return_value = {"metadatas": [None, {}, {"source_id": "CUSTOM-DEMO"}]}
        self.assertEqual(_add_missing_demo_documents(store), 15)
        store.delete.assert_not_called()

    def test_new_topics_are_retrievable_in_sparse_top_three(self):
        retriever = BM25Retriever.from_documents(mock_guidelines, preprocess_func=_tokenize_for_bm25)
        retriever.k = 3
        for query, expected in (("咽痛 喉咙痛 鼻塞 流涕", "DEMO-ENT-001"),
                                ("尿频 尿急 尿痛 排尿不适", "DEMO-URO-001"),
                                ("皮疹 瘙痒 皮肤发红", "DEMO-DERM-001"),
                                ("腰痛 下背痛 疼痛放射", "DEMO-MSK-001")):
            with self.subTest(query=query):
                self.assertIn(expected, [doc.metadata["source_id"] for doc in retriever.invoke(query)])


if __name__ == "__main__":
    unittest.main()
