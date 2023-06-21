package gobert_test

import (
	"fmt"
	"path/filepath"

	. "github.com/go-skynet/go-bert.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("gobert binding", func() {
	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := New("not-existing")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
	Context("Embedding", func() {
		It("get embeddings", func() {
			model, err := New(filepath.Join("fixtures", "model.bin"))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			embeddings, err := model.StringEmbeddings("foo")
			Expect(err).ToNot(HaveOccurred())
			fmt.Println(embeddings)
		})
	})
})
