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
	"unsafe"
)

type Bert struct {
	state unsafe.Pointer
}

func New(model string) (*Bert, error) {
	state := C.bert_allocate_state()
	modelPath := C.CString(model)
	result := C.bert_bootstrap(modelPath, state)
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &Bert{state: state}, nil
}

func (l *Bert) Embeddings(text string, opts ...PredictOption) ([]float32, error) {

	po := NewPredictOptions(opts...)
	embeddings := make([]float32, po.EmbeddingSize)
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

func (l *Bert) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	po := NewPredictOptions(opts...)
	embeddings := make([]float32, po.EmbeddingSize)
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
