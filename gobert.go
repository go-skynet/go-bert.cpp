package gobert

// #cgo CFLAGS: -I./bert.cpp/ggml/include/ggml/ -I./bert.cpp/ggml/src/ -I./bert.cpp
// #cgo CXXFLAGS: -I./bert.cpp/ggml/include/ggml/ -I./bert.cpp/ggml/src/ -I./bert.cpp
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -lgobert -lm -lstdc++
// #include <stdlib.h>
// #include <gobert.h>
import "C"
import (
	"fmt"
	common "github.com/go-skynet/go-common"
	"unsafe"
)

type Bert struct {
	state unsafe.Pointer
}

func New(modelPath string) (*Bert, error) {
	return NewWithInitializationOptions(modelPath, DefaultModelInitializationOptions)
}

// TODO: Not sure about this one. This exists to be an analog of the other go-* backends, but we ignore every option...
// May not actually be useful since go interfaces apply to functions with recievers - will this ever get used?
func NewWithInitializationOptions(modelPath string, _ common.InitializationOptions) (*Bert, error) {
	state := C.bert_allocate_state()
	cModelPath := C.CString(modelPath)
	result := C.bert_bootstrap(cModelPath, state)
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &Bert{state: state}, nil
}

func (l *Bert) StringEmbeddings(text string, opts ...common.PredictTextOptionSetter) ([]float32, error) {
	return l.StringEmbeddingsWithOptions(text, *MergePredictOptionsWithDefaults(opts...))
}

func (l *Bert) StringEmbeddingsWithOptions(text string, po common.PredictTextOptions) ([]float32, error) {
	embeddings := make([]float32, po.Tokens)
	embeddingsPtr := (*C.float)(&embeddings[0])

	params := C.bert_allocate_params(C.CString(text), C.int(po.Threads))
	ret := C.bert_embeddings(params, l.state, embeddingsPtr)
	if ret != 0 {
		return []float32{}, fmt.Errorf("inference failed")
	}

	C.bert_free_params(params)

	// Remove trailing 0s
	for i := len(embeddings) - 1; i >= 0; i-- {
		if embeddings[i] == 0.0 {
			embeddings = embeddings[:i]
		} else {
			break
		}
	}

	return embeddings, nil
}

func (l *Bert) TokenEmbeddings(tokens []int, opts ...common.PredictTextOptionSetter) ([]float32, error) {
	return l.TokenEmbeddingsWithOptions(tokens, *MergePredictOptionsWithDefaults(opts...))
}

func (l *Bert) TokenEmbeddingsWithOptions(tokens []int, po common.PredictTextOptions) ([]float32, error) {
	embeddings := make([]float32, po.Tokens)
	embeddingsPtr := (*C.float)(&embeddings[0])

	myArray := (*C.int)(C.malloc(C.size_t(len(tokens)) * C.sizeof_int))

	// Copy the values from the Go slice to the C array
	for i, v := range tokens {
		(*[1<<31 - 1]int32)(unsafe.Pointer(myArray))[i] = int32(v)
	}

	params := C.bert_allocate_params(C.CString(""), C.int(po.Threads))
	ret := C.bert_token_embeddings(params, l.state, myArray, C.int(len(tokens)), embeddingsPtr)
	if ret != 0 {
		return []float32{}, fmt.Errorf("inference failed")
	}

	C.bert_free_params(params)

	// Remove trailing 0s
	for i := len(embeddings) - 1; i >= 0; i-- {
		if embeddings[i] == 0.0 {
			embeddings = embeddings[:i]
		} else {
			break
		}
	}

	return embeddings, nil
}

func (l *Bert) Free() {
	C.bert_free_model(l.state)
}
