package gobert

import (
	common "github.com/go-skynet/go-common"
)

var DefaultPredictOptions common.PredictTextOptions = common.PredictTextOptions{
	Threads: 4,
	Tokens:  99999, // Formerly EmbeddingSize, llama embeddings uses Tokens for this. @mudler, do we like this convention, or should we seperate the embedding token size for some reason?
}

var MergePredictOptionsWithDefaults = common.GetMergePredictTextOptionsFnFromDefault(DefaultPredictOptions)
