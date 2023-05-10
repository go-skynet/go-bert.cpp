package gobert_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestGPT(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "go-gobert test suite")
}
