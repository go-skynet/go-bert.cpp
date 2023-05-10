package gobert

type PredictOptions struct {
	Threads       int
	EmbeddingSize int
}

type PredictOption func(p *PredictOptions)

var DefaultOptions PredictOptions = PredictOptions{
	Threads:       4,
	EmbeddingSize: 99999,
}

// SetThreads sets the number of threads to use for text generation.
func SetEmbeddingSize(es int) PredictOption {
	return func(p *PredictOptions) {
		p.EmbeddingSize = es
	}
}

// SetThreads sets the number of threads to use for text generation.
func SetThreads(threads int) PredictOption {
	return func(p *PredictOptions) {
		p.Threads = threads
	}
}

// Create a new PredictOptions object with the given options.
func NewPredictOptions(opts ...PredictOption) PredictOptions {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}
