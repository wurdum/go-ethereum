// Copyright 2014 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package types

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math/big"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/rlp"
)

var (
	TraceShowStacktrace      bool
	TraceShowArbOSRead       bool
	TraceShowDeepstate       bool
	TraceShowBurn            bool
	TraceShowOpcodes         bool
	TraceShowStateRootChange bool

	targetBlockNumber  int64
	currentBlockNumber int64
	transactionIndex   int64
)

func init() {
	TraceShowStacktrace = os.Getenv("TRACE_SHOW_STACKTRACE") == "true"
	TraceShowArbOSRead = os.Getenv("TRACE_SHOW_ARBOS_READ") == "true"
	TraceShowDeepstate = os.Getenv("TRACE_SHOW_DEEPSTATE") == "true"
	TraceShowBurn = os.Getenv("TRACE_SHOW_BURN") == "true"
	TraceShowOpcodes = os.Getenv("TRACE_SHOW_OPCODES") == "true"
	TraceShowStateRootChange = os.Getenv("TRACE_SHOW_STATE_ROOT_CHANGE") == "true"

	if val, err := strconv.ParseInt(os.Getenv("TARGET_BLOCK_NUMBER"), 10, 64); err == nil {
		targetBlockNumber = val
	} else {
		targetBlockNumber = -1
	}
}

func GetTargetBlockNumber() int64 {
	return targetBlockNumber
}

func SetCurrentBlockNumber(blockNumber int64) {
	currentBlockNumber = blockNumber
}

func SetTransactionIndex(txIndex int64) {
	transactionIndex = txIndex
}

func IsTargetBlock() bool {
	return currentBlockNumber == targetBlockNumber
}

func OLogAlways(log string) {
	if TraceShowStacktrace {
		println(GetCallStackString())
	}

	println(log)
}

func OLog2(log string) {
	if !IsTargetBlock() {
		return
	}

	if TraceShowStacktrace {
		println(fmt.Sprintf("b=%d, t=%d %s", currentBlockNumber, transactionIndex, GetCallStackString()))
	}

	println(fmt.Sprintf("b=%d, t=%d %s", currentBlockNumber, transactionIndex, log))
}

func OLog2Fast(log string) {
	if !IsTargetBlock() {
		return
	}

	println(fmt.Sprintf("b=%d, t=%d %s", currentBlockNumber, transactionIndex, log))
}

func OLog(scope, key, value string) {
	if !IsTargetBlock() {
		return
	}

	if TraceShowStacktrace {
		println(fmt.Sprintf("b=%d, t=%d %s", currentBlockNumber, transactionIndex, GetCallStackString()))
	}

	println(fmt.Sprintf("b=%d, t=%d, s=%s: %s %s", currentBlockNumber, transactionIndex, scope, key, value))
}

func Log3(block *Block) {
	if !IsTargetBlock() {
		return
	}

	// Check if block is nil
	if block == nil {
		return
	}

	// Start building the output
	var output strings.Builder
	header := block.Header()

	// Block header section
	output.WriteString(fmt.Sprintf("Block %d\n", header.Number.Uint64()))
	output.WriteString("  Header:\n")
	output.WriteString(fmt.Sprintf("    Hash: %s\n", block.Hash().Hex()))
	output.WriteString(fmt.Sprintf("    Number: %d\n", header.Number.Uint64()))
	output.WriteString(fmt.Sprintf("    Parent: %s\n", header.ParentHash.Hex()))
	output.WriteString(fmt.Sprintf("    Beneficiary: %s\n", strings.ToLower(header.Coinbase.Hex())))
	output.WriteString(fmt.Sprintf("    Gas Limit: %d\n", header.GasLimit))
	output.WriteString(fmt.Sprintf("    Gas Used: %d\n", header.GasUsed))
	output.WriteString(fmt.Sprintf("    Timestamp: %d\n", header.Time))
	output.WriteString(fmt.Sprintf("    Extra Data: %x\n", header.Extra))
	output.WriteString(fmt.Sprintf("    Difficulty: %s\n", header.Difficulty.String()))
	output.WriteString(fmt.Sprintf("    Mix Hash: %s\n", header.MixDigest.Hex()))
	output.WriteString(fmt.Sprintf("    Nonce: %d\n", header.Nonce.Uint64()))
	output.WriteString(fmt.Sprintf("    Uncles Hash: %s\n", header.UncleHash.Hex()))
	output.WriteString(fmt.Sprintf("    Tx Root: %s\n", header.TxHash.Hex()))
	output.WriteString(fmt.Sprintf("    Receipts Root: %s\n", header.ReceiptHash.Hex()))
	output.WriteString(fmt.Sprintf("    State Root: %s\n", header.Root.Hex()))
	if header.BaseFee != nil {
		output.WriteString(fmt.Sprintf("    BaseFeePerGas: %s\n", header.BaseFee.String()))
	} else {
		output.WriteString("    BaseFeePerGas: <nil>\n")
	}

	// IsPostMerge check
	isPostMerge := header.Difficulty.Cmp(big.NewInt(0)) == 0 && header.MixDigest == (common.Hash{})
	output.WriteString(fmt.Sprintf("    IsPostMerge: %t\n", isPostMerge))

	// TotalDifficulty - we'll use block number + 1 as approximation
	totalDifficulty := new(big.Int).Add(header.Number, big.NewInt(1))
	output.WriteString(fmt.Sprintf("    TotalDifficulty: %s\n", totalDifficulty.String()))

	// Uncles section
	output.WriteString("  Uncles:\n")
	uncles := block.Uncles()
	if len(uncles) == 0 {
		// Empty uncles section
	} else {
		for _, uncle := range uncles {
			output.WriteString(fmt.Sprintf("    Uncle Hash: %s\n", uncle.Hash().Hex()))
		}
	}

	// Transactions section
	output.WriteString("  Transactions:\n")
	txs := block.Transactions()
	for _, tx := range txs {
		output.WriteString(fmt.Sprintf("    Hash:      %s\n", tx.Hash().Hex()))

		// Get from address
		from, err := Sender(LatestSignerForChainID(tx.ChainId()), tx)
		if err != nil {
			output.WriteString(fmt.Sprintf("    From:      <error: %v>\n", err))
		} else {
			output.WriteString(fmt.Sprintf("    From:      %s\n", strings.ToLower(from.Hex())))
		}

		// To address
		if tx.To() != nil {
			output.WriteString(fmt.Sprintf("    To:        %s\n", strings.ToLower(tx.To().Hex())))
		} else {
			output.WriteString("    To:        <nil>\n")
		}

		// Transaction type
		txType := tx.Type()
		txTypeName := "Unknown"
		switch txType {
		case LegacyTxType:
			txTypeName = "Legacy"
		case AccessListTxType:
			txTypeName = "AccessList"
		case DynamicFeeTxType:
			txTypeName = "EIP1559"
		case BlobTxType:
			txTypeName = "Blob"
		case SetCodeTxType:
			txTypeName = "SetCode"
		default:
			txTypeName = fmt.Sprintf("%d", txType)
		}
		output.WriteString(fmt.Sprintf("    TxType:    %s\n", txTypeName))

		// Gas fees
		if tx.GasTipCap() != nil {
			output.WriteString(fmt.Sprintf("    MaxPriorityFeePerGas: %s\n", tx.GasTipCap().String()))
		} else {
			output.WriteString("    MaxPriorityFeePerGas: 0\n")
		}
		if tx.GasFeeCap() != nil {
			output.WriteString(fmt.Sprintf("    MaxFeePerGas: %s\n", tx.GasFeeCap().String()))
		} else if tx.GasPrice() != nil {
			output.WriteString(fmt.Sprintf("    MaxFeePerGas: %s\n", tx.GasPrice().String()))
		} else {
			output.WriteString("    MaxFeePerGas: 0\n")
		}

		// Additional fields
		output.WriteString("    SourceHash:Ignore\n")
		output.WriteString("    Mint:      Ignore\n")
		output.WriteString("    OpSystem:  False\n")
		output.WriteString(fmt.Sprintf("    Gas Limit: %d\n", tx.Gas()))
		output.WriteString(fmt.Sprintf("    Nonce:     %d\n", tx.Nonce()))
		output.WriteString(fmt.Sprintf("    Value:     %s\n", tx.Value().String()))

		// Data field
		data := tx.Data()
		if len(data) > 0 {
			output.WriteString(fmt.Sprintf("    Data:      %x\n", data))
		} else {
			output.WriteString("    Data:      \n")
		}

		// Signature
		v, r, s := tx.RawSignatureValues()
		if r != nil && s != nil {
			// Combine R and S for signature
			sigBytes := make([]byte, 64)
			r.FillBytes(sigBytes[:32])
			s.FillBytes(sigBytes[32:])
			output.WriteString(fmt.Sprintf("    Signature: %x\n", sigBytes))
		} else {
			output.WriteString("    Signature:\n")
		}

		// V value
		if v != nil {
			output.WriteString(fmt.Sprintf("    V:         %s\n", v.String()))
		} else {
			output.WriteString("    V:\n")
		}

		// ChainId
		if tx.ChainId() != nil {
			output.WriteString(fmt.Sprintf("    ChainId:   %s\n", tx.ChainId().String()))
		} else {
			output.WriteString("    ChainId:\n")
		}

		// Timestamp (usually 0 for regular transactions)
		output.WriteString("    Timestamp: 0\n")
	}

	// Withdrawals section
	output.WriteString("  Withdrawals:\n")
	withdrawals := block.Withdrawals()
	if withdrawals != nil && len(withdrawals) > 0 {
		for _, w := range withdrawals {
			output.WriteString(fmt.Sprintf("    Index: %d, Validator: %d, Address: %s, Amount: %d\n",
				w.Index, w.Validator, w.Address.Hex(), w.Amount))
		}
	}

	// Print the complete output
	println(output.String())
}

func GetCallStackString() string {
	frames := GetCallStack()
	if len(frames) == 0 {
		return ""
	}

	// Pre-allocate with reasonable capacity to avoid reallocations
	var b strings.Builder
	b.Grow(len(frames) * 30) // Rough estimate

	for i, frame := range frames {
		if i > 0 {
			b.WriteString(" → ")
		}

		// Format: file:line [StructName.]MethodName
		b.WriteString(frame.File)
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(frame.LineNumber))
		b.WriteByte(' ')

		if frame.StructureName != "" {
			b.WriteString(frame.StructureName)
			b.WriteByte('.')
		}
		b.WriteString(frame.MethodName)
	}

	return b.String()
}

type StackFrame struct {
	File          string `json:"file"`
	LineNumber    int    `json:"lineNumber"`
	StructureName string `json:"structureName"`
	MethodName    string `json:"methodName"`
}

func IsNitroCall(fileName string) bool {
	return strings.Contains(fileName, "nitro") && strings.HasSuffix(fileName, ".go")
}

// GetCallStack returns a structured representation of the call stack
func GetCallStack() []StackFrame {
	// Skip this function and the calling function
	const skip = 2
	const depth = 20

	// Create a buffer for program counters
	var pcs [depth]uintptr
	n := runtime.Callers(skip, pcs[:])

	if n == 0 {
		return []StackFrame{}
	}

	// Get frame information
	frames := runtime.CallersFrames(pcs[:n])

	// Build the call stack
	var stackFrames []StackFrame
	for {
		frame, more := frames.Next()

		// Parse the function name to extract package, type (if any), and method
		funcNameParts := strings.Split(frame.Function, ".")

		var structureName, methodName string

		if len(funcNameParts) >= 2 {
			// Check if this is a method on a struct type
			methodName = funcNameParts[len(funcNameParts)-1]

			// Look for struct type pattern
			typeIndex := len(funcNameParts) - 2
			typePart := funcNameParts[typeIndex]

			// Check if it's a struct method (has a type component)
			if strings.Contains(typePart, ")") ||
				(typePart != "" && typePart[0] >= 'A' && typePart[0] <= 'Z') {
				// Clean up the type name (remove pointer symbol)
				typePart = strings.TrimPrefix(typePart, "(*")
				typePart = strings.TrimSuffix(typePart, ")")

				if typePart != "" {
					// It's a method on a struct
					structureName = typePart
				}
			} else {
				// Regular function
				structureName = "" // No structure for regular functions
			}
		} else if len(funcNameParts) == 1 {
			methodName = funcNameParts[0]
			structureName = ""
		} else {
			methodName = frame.Function
			structureName = ""
		}

		// Extract filename without full path
		fileName := frame.File
		if lastSlash := strings.LastIndexByte(fileName, '/'); lastSlash >= 0 {
			fileName = fileName[lastSlash+1:]
		}

		if IsNitroCall(frame.File) {
			stackFrame := StackFrame{
				File:          fileName,
				LineNumber:    frame.Line,
				StructureName: structureName,
				MethodName:    methodName,
			}

			stackFrames = append(stackFrames, stackFrame)
		}

		if !more {
			break
		}
	}

	// Reverse the call stack to show in chronological order
	for i, j := 0, len(stackFrames)-1; i < j; i, j = i+1, j-1 {
		stackFrames[i], stackFrames[j] = stackFrames[j], stackFrames[i]
	}

	return stackFrames
}

var (
	ErrInvalidSig           = errors.New("invalid transaction v, r, s values")
	ErrUnexpectedProtection = errors.New("transaction type does not supported EIP-155 protected signatures")
	ErrInvalidTxType        = errors.New("transaction type not valid in this context")
	ErrTxTypeNotSupported   = errors.New("transaction type not supported")
	ErrGasFeeCapTooLow      = errors.New("fee cap less than base fee")
	errShortTypedTx         = errors.New("typed transaction too short")
	errInvalidYParity       = errors.New("'yParity' field must be 0 or 1")
	errVYParityMismatch     = errors.New("'v' and 'yParity' fields do not match")
	errVYParityMissing      = errors.New("missing 'yParity' or 'v' field in transaction")
)

// Transaction types.
const (
	ArbitrumDepositTxType         = 0x64
	ArbitrumUnsignedTxType        = 0x65
	ArbitrumContractTxType        = 0x66
	ArbitrumRetryTxType           = 0x68
	ArbitrumSubmitRetryableTxType = 0x69
	ArbitrumInternalTxType        = 0x6A
	ArbitrumLegacyTxType          = 0x78

	LegacyTxType     = 0x00
	AccessListTxType = 0x01
	DynamicFeeTxType = 0x02
	BlobTxType       = 0x03
	SetCodeTxType    = 0x04
)

// Transaction is an Ethereum transaction.
type Transaction struct {
	inner TxData    // Consensus contents of a transaction
	time  time.Time // Time first seen locally (spam avoidance)

	// Arbitrum cache of the calldata units at a brotli compression level.
	// The top 8 bits are the brotli compression level last used to compute this,
	// and the remaining 56 bits are the calldata units at that compression level.
	calldataUnitsForBrotliCompressionLevel atomic.Uint64

	// caches
	hash atomic.Pointer[common.Hash]
	size atomic.Uint64
	from atomic.Pointer[sigCache]
}

// GetRawCachedCalldataUnits returns the cached brotli compression level and corresponding calldata units,
// or (0, 0) if the cache is empty.
func (tx *Transaction) GetRawCachedCalldataUnits() (uint64, uint64) {
	repr := tx.calldataUnitsForBrotliCompressionLevel.Load()
	cachedCompressionLevel := repr >> 56
	calldataUnits := repr & ((1 << 56) - 1)
	return cachedCompressionLevel, calldataUnits
}

// GetCachedCalldataUnits returns the cached calldata units for a given brotli compression level,
// returning nil if no cache is present or the cache is for a different compression level.
func (tx *Transaction) GetCachedCalldataUnits(requestedCompressionLevel uint64) *uint64 {
	cachedCompressionLevel, cachedUnits := tx.GetRawCachedCalldataUnits()
	if IsTargetBlock() {
		OLog2(fmt.Sprintf("level=%d units=%d", requestedCompressionLevel, cachedUnits))
	}
	if cachedUnits == 0 {
		// empty cache
		return nil
	}
	if cachedCompressionLevel != requestedCompressionLevel {
		// wrong compression level
		return nil
	}
	return &cachedUnits
}

// SetCachedCalldataUnits sets the cached brotli compression level and corresponding calldata units,
// or clears the cache if the values are too large to fit (at least 2**8 and 2**56 respectively).
// Note that a zero calldataUnits is also treated as an empty cache.
func (tx *Transaction) SetCachedCalldataUnits(compressionLevel uint64, calldataUnits uint64) {
	var repr uint64
	// Ensure the compressionLevel and calldataUnits will fit.
	// Otherwise, just clear the cache.
	if compressionLevel < 1<<8 && calldataUnits < 1<<56 {
		repr = compressionLevel<<56 | calldataUnits
	}
	tx.calldataUnitsForBrotliCompressionLevel.Store(repr)
}

// NewTx creates a new transaction.
func NewTx(inner TxData) *Transaction {
	tx := new(Transaction)
	tx.setDecoded(inner.copy(), 0)
	return tx
}

// TxData is the underlying data of a transaction.
//
// This is implemented by DynamicFeeTx, LegacyTx and AccessListTx.
type TxData interface {
	txType() byte // returns the type ID
	copy() TxData // creates a deep copy and initializes all fields

	chainID() *big.Int
	accessList() AccessList
	data() []byte
	gas() uint64
	gasPrice() *big.Int
	gasTipCap() *big.Int
	gasFeeCap() *big.Int
	value() *big.Int
	nonce() uint64
	to() *common.Address

	rawSignatureValues() (v, r, s *big.Int)
	setSignatureValues(chainID, v, r, s *big.Int)

	skipNonceChecks() bool
	skipFromEOACheck() bool

	// effectiveGasPrice computes the gas price paid by the transaction, given
	// the inclusion block baseFee.
	//
	// Unlike other TxData methods, the returned *big.Int should be an independent
	// copy of the computed value, i.e. callers are allowed to mutate the result.
	// Method implementations can use 'dst' to store the result.
	effectiveGasPrice(dst *big.Int, baseFee *big.Int) *big.Int

	encode(*bytes.Buffer) error
	decode([]byte) error

	// sigHash returns the hash of the transaction that is ought to be signed
	sigHash(*big.Int) common.Hash
}

// EncodeRLP implements rlp.Encoder
func (tx *Transaction) EncodeRLP(w io.Writer) error {
	if tx.Type() == LegacyTxType {
		return rlp.Encode(w, tx.inner)
	}
	// It's an EIP-2718 typed TX envelope.
	buf := encodeBufferPool.Get().(*bytes.Buffer)
	defer encodeBufferPool.Put(buf)
	buf.Reset()
	if err := tx.encodeTyped(buf); err != nil {
		return err
	}
	return rlp.Encode(w, buf.Bytes())
}

// encodeTyped writes the canonical encoding of a typed transaction to w.
func (tx *Transaction) encodeTyped(w *bytes.Buffer) error {
	w.WriteByte(tx.Type())
	return tx.inner.encode(w)
}

// MarshalBinary returns the canonical encoding of the transaction.
// For legacy transactions, it returns the RLP encoding. For EIP-2718 typed
// transactions, it returns the type and payload.
func (tx *Transaction) MarshalBinary() ([]byte, error) {
	if tx.Type() == LegacyTxType {
		return rlp.EncodeToBytes(tx.inner)
	}
	var buf bytes.Buffer
	err := tx.encodeTyped(&buf)
	return buf.Bytes(), err
}

// DecodeRLP implements rlp.Decoder
func (tx *Transaction) DecodeRLP(s *rlp.Stream) error {
	kind, size, err := s.Kind()
	switch {
	case err != nil:
		return err
	case kind == rlp.List:
		// It's a legacy transaction.
		var inner LegacyTx
		err := s.Decode(&inner)
		if err == nil {
			tx.setDecoded(&inner, rlp.ListSize(size))
		}
		return err
	case kind == rlp.Byte:
		return errShortTypedTx
	default:
		// It's an EIP-2718 typed TX envelope.
		// First read the tx payload bytes into a temporary buffer.
		b, buf, err := getPooledBuffer(size)
		if err != nil {
			return err
		}
		defer encodeBufferPool.Put(buf)
		if err := s.ReadBytes(b); err != nil {
			return err
		}
		// Now decode the inner transaction.
		inner, err := tx.decodeTyped(b, true)
		if err == nil {
			tx.setDecoded(inner, size)
		}
		return err
	}
}

// UnmarshalBinary decodes the canonical encoding of transactions.
// It supports legacy RLP transactions and EIP-2718 typed transactions.
func (tx *Transaction) UnmarshalBinary(b []byte) error {
	if len(b) > 0 && b[0] > 0x7f {
		// It's a legacy transaction.
		var data LegacyTx
		err := rlp.DecodeBytes(b, &data)
		if err != nil {
			return err
		}
		tx.setDecoded(&data, uint64(len(b)))
		return nil
	}
	// It's an EIP-2718 typed transaction envelope.
	inner, err := tx.decodeTyped(b, false)
	if err != nil {
		return err
	}
	tx.setDecoded(inner, uint64(len(b)))
	return nil
}

// decodeTyped decodes a typed transaction from the canonical format.
func (tx *Transaction) decodeTyped(b []byte, arbParsing bool) (TxData, error) {
	if len(b) <= 1 {
		return nil, errShortTypedTx
	}
	var inner TxData
	if arbParsing {
		switch b[0] {
		case ArbitrumDepositTxType:
			inner = new(ArbitrumDepositTx)
		case ArbitrumInternalTxType:
			inner = new(ArbitrumInternalTx)
		case ArbitrumUnsignedTxType:
			inner = new(ArbitrumUnsignedTx)
		case ArbitrumContractTxType:
			inner = new(ArbitrumContractTx)
		case ArbitrumRetryTxType:
			inner = new(ArbitrumRetryTx)
		case ArbitrumSubmitRetryableTxType:
			inner = new(ArbitrumSubmitRetryableTx)
		case ArbitrumLegacyTxType:
			inner = new(ArbitrumLegacyTxData)
		default:
			arbParsing = false
		}
	}
	if !arbParsing {
		switch b[0] {
		case AccessListTxType:
			inner = new(AccessListTx)
		case DynamicFeeTxType:
			inner = new(DynamicFeeTx)
		case BlobTxType:
			inner = new(BlobTx)
		case SetCodeTxType:
			inner = new(SetCodeTx)
		default:
			return nil, ErrTxTypeNotSupported
		}
	}
	err := inner.decode(b[1:])
	return inner, err
}

// setDecoded sets the inner transaction and size after decoding.
func (tx *Transaction) setDecoded(inner TxData, size uint64) {
	tx.inner = inner
	tx.time = time.Now()
	if size > 0 {
		tx.size.Store(size)
	}
}

func sanityCheckSignature(v *big.Int, r *big.Int, s *big.Int, maybeProtected bool) error {
	if isProtectedV(v) && !maybeProtected {
		return ErrUnexpectedProtection
	}

	var plainV byte
	if isProtectedV(v) {
		chainID := deriveChainId(v).Uint64()
		plainV = byte(v.Uint64() - 35 - 2*chainID)
	} else if maybeProtected {
		// Only EIP-155 signatures can be optionally protected. Since
		// we determined this v value is not protected, it must be a
		// raw 27 or 28.
		plainV = byte(v.Uint64() - 27)
	} else {
		// If the signature is not optionally protected, we assume it
		// must already be equal to the recovery id.
		plainV = byte(v.Uint64())
	}
	if !crypto.ValidateSignatureValues(plainV, r, s, false) {
		return ErrInvalidSig
	}

	return nil
}

func isProtectedV(V *big.Int) bool {
	if V.BitLen() <= 8 {
		v := V.Uint64()
		return v != 27 && v != 28 && v != 1 && v != 0
	}
	// anything not 27 or 28 is considered protected
	return true
}

// Protected says whether the transaction is replay-protected.
func (tx *Transaction) Protected() bool {
	switch tx := tx.inner.(type) {
	case *LegacyTx:
		return tx.V != nil && isProtectedV(tx.V)
	default:
		return true
	}
}

// Type returns the transaction type.
func (tx *Transaction) Type() uint8 {
	return tx.inner.txType()
}

func (tx *Transaction) GetInner() TxData {
	return tx.inner.copy()
}

// ChainId returns the EIP155 chain ID of the transaction. The return value will always be
// non-nil. For legacy transactions which are not replay-protected, the return value is
// zero.
func (tx *Transaction) ChainId() *big.Int {
	return tx.inner.chainID()
}

// Data returns the input data of the transaction.
func (tx *Transaction) Data() []byte { return tx.inner.data() }

// AccessList returns the access list of the transaction.
func (tx *Transaction) AccessList() AccessList { return tx.inner.accessList() }

// Gas returns the gas limit of the transaction.
func (tx *Transaction) Gas() uint64 { return tx.inner.gas() }

// GasPrice returns the gas price of the transaction.
func (tx *Transaction) GasPrice() *big.Int { return new(big.Int).Set(tx.inner.gasPrice()) }

// GasTipCap returns the gasTipCap per gas of the transaction.
func (tx *Transaction) GasTipCap() *big.Int { return new(big.Int).Set(tx.inner.gasTipCap()) }

// GasFeeCap returns the fee cap per gas of the transaction.
func (tx *Transaction) GasFeeCap() *big.Int { return new(big.Int).Set(tx.inner.gasFeeCap()) }

// Value returns the ether amount of the transaction.
func (tx *Transaction) Value() *big.Int { return new(big.Int).Set(tx.inner.value()) }

// Nonce returns the sender account nonce of the transaction.
func (tx *Transaction) Nonce() uint64 { return tx.inner.nonce() }

// To returns the recipient address of the transaction.
// For contract-creation transactions, To returns nil.
func (tx *Transaction) To() *common.Address {
	return copyAddressPtr(tx.inner.to())
}

// Cost returns (gas * gasPrice) + (blobGas * blobGasPrice) + value.
func (tx *Transaction) Cost() *big.Int {
	total := new(big.Int).Mul(tx.GasPrice(), new(big.Int).SetUint64(tx.Gas()))
	if tx.Type() == BlobTxType {
		total.Add(total, new(big.Int).Mul(tx.BlobGasFeeCap(), new(big.Int).SetUint64(tx.BlobGas())))
	}
	total.Add(total, tx.Value())
	return total
}

// RawSignatureValues returns the V, R, S signature values of the transaction.
// The return values should not be modified by the caller.
// The return values may be nil or zero, if the transaction is unsigned.
func (tx *Transaction) RawSignatureValues() (v, r, s *big.Int) {
	return tx.inner.rawSignatureValues()
}

// GasFeeCapCmp compares the fee cap of two transactions.
func (tx *Transaction) GasFeeCapCmp(other *Transaction) int {
	return tx.inner.gasFeeCap().Cmp(other.inner.gasFeeCap())
}

// GasFeeCapIntCmp compares the fee cap of the transaction against the given fee cap.
func (tx *Transaction) GasFeeCapIntCmp(other *big.Int) int {
	return tx.inner.gasFeeCap().Cmp(other)
}

// GasTipCapCmp compares the gasTipCap of two transactions.
func (tx *Transaction) GasTipCapCmp(other *Transaction) int {
	return tx.inner.gasTipCap().Cmp(other.inner.gasTipCap())
}

// GasTipCapIntCmp compares the gasTipCap of the transaction against the given gasTipCap.
func (tx *Transaction) GasTipCapIntCmp(other *big.Int) int {
	return tx.inner.gasTipCap().Cmp(other)
}

// EffectiveGasTip returns the effective miner gasTipCap for the given base fee.
// Note: if the effective gasTipCap is negative, this method returns both error
// the actual negative value, _and_ ErrGasFeeCapTooLow
func (tx *Transaction) EffectiveGasTip(baseFee *big.Int) (*big.Int, error) {
	if baseFee == nil {
		return tx.GasTipCap(), nil
	}
	var err error
	gasFeeCap := tx.GasFeeCap()
	if gasFeeCap.Cmp(baseFee) < 0 {
		err = ErrGasFeeCapTooLow
	}
	gasFeeCap = gasFeeCap.Sub(gasFeeCap, baseFee)

	gasTipCap := tx.GasTipCap()
	if gasTipCap.Cmp(gasFeeCap) < 0 {
		return gasTipCap, err
	}
	return gasFeeCap, err
}

// EffectiveGasTipValue is identical to EffectiveGasTip, but does not return an
// error in case the effective gasTipCap is negative
func (tx *Transaction) EffectiveGasTipValue(baseFee *big.Int) *big.Int {
	effectiveTip, _ := tx.EffectiveGasTip(baseFee)
	return effectiveTip
}

// EffectiveGasTipCmp compares the effective gasTipCap of two transactions assuming the given base fee.
func (tx *Transaction) EffectiveGasTipCmp(other *Transaction, baseFee *big.Int) int {
	if baseFee == nil {
		return tx.GasTipCapCmp(other)
	}
	return tx.EffectiveGasTipValue(baseFee).Cmp(other.EffectiveGasTipValue(baseFee))
}

// EffectiveGasTipIntCmp compares the effective gasTipCap of a transaction to the given gasTipCap.
func (tx *Transaction) EffectiveGasTipIntCmp(other *big.Int, baseFee *big.Int) int {
	if baseFee == nil {
		return tx.GasTipCapIntCmp(other)
	}
	return tx.EffectiveGasTipValue(baseFee).Cmp(other)
}

// BlobGas returns the blob gas limit of the transaction for blob transactions, 0 otherwise.
func (tx *Transaction) BlobGas() uint64 {
	if blobtx, ok := tx.inner.(*BlobTx); ok {
		return blobtx.blobGas()
	}
	return 0
}

// BlobGasFeeCap returns the blob gas fee cap per blob gas of the transaction for blob transactions, nil otherwise.
func (tx *Transaction) BlobGasFeeCap() *big.Int {
	if blobtx, ok := tx.inner.(*BlobTx); ok {
		return blobtx.BlobFeeCap.ToBig()
	}
	return nil
}

// BlobHashes returns the hashes of the blob commitments for blob transactions, nil otherwise.
func (tx *Transaction) BlobHashes() []common.Hash {
	if blobtx, ok := tx.inner.(*BlobTx); ok {
		return blobtx.BlobHashes
	}
	return nil
}

// BlobTxSidecar returns the sidecar of a blob transaction, nil otherwise.
func (tx *Transaction) BlobTxSidecar() *BlobTxSidecar {
	if blobtx, ok := tx.inner.(*BlobTx); ok {
		return blobtx.Sidecar
	}
	return nil
}

// BlobGasFeeCapCmp compares the blob fee cap of two transactions.
func (tx *Transaction) BlobGasFeeCapCmp(other *Transaction) int {
	return tx.BlobGasFeeCap().Cmp(other.BlobGasFeeCap())
}

// BlobGasFeeCapIntCmp compares the blob fee cap of the transaction against the given blob fee cap.
func (tx *Transaction) BlobGasFeeCapIntCmp(other *big.Int) int {
	return tx.BlobGasFeeCap().Cmp(other)
}

// WithoutBlobTxSidecar returns a copy of tx with the blob sidecar removed.
func (tx *Transaction) WithoutBlobTxSidecar() *Transaction {
	blobtx, ok := tx.inner.(*BlobTx)
	if !ok {
		return tx
	}
	cpy := &Transaction{
		inner: blobtx.withoutSidecar(),
		time:  tx.time,
	}
	// Note: tx.size cache not carried over because the sidecar is included in size!
	if h := tx.hash.Load(); h != nil {
		cpy.hash.Store(h)
	}
	if f := tx.from.Load(); f != nil {
		cpy.from.Store(f)
	}
	return cpy
}

// WithBlobTxSidecar returns a copy of tx with the blob sidecar added.
func (tx *Transaction) WithBlobTxSidecar(sideCar *BlobTxSidecar) *Transaction {
	blobtx, ok := tx.inner.(*BlobTx)
	if !ok {
		return tx
	}
	cpy := &Transaction{
		inner: blobtx.withSidecar(sideCar),
		time:  tx.time,
	}
	// Note: tx.size cache not carried over because the sidecar is included in size!
	if h := tx.hash.Load(); h != nil {
		cpy.hash.Store(h)
	}
	if f := tx.from.Load(); f != nil {
		cpy.from.Store(f)
	}
	return cpy
}

// SetCodeAuthorizations returns the authorizations list of the transaction.
func (tx *Transaction) SetCodeAuthorizations() []SetCodeAuthorization {
	setcodetx, ok := tx.inner.(*SetCodeTx)
	if !ok {
		return nil
	}
	return setcodetx.AuthList
}

// SetCodeAuthorities returns a list of unique authorities from the
// authorization list.
func (tx *Transaction) SetCodeAuthorities() []common.Address {
	setcodetx, ok := tx.inner.(*SetCodeTx)
	if !ok {
		return nil
	}
	var (
		marks = make(map[common.Address]bool)
		auths = make([]common.Address, 0, len(setcodetx.AuthList))
	)
	for _, auth := range setcodetx.AuthList {
		if addr, err := auth.Authority(); err == nil {
			if marks[addr] {
				continue
			}
			marks[addr] = true
			auths = append(auths, addr)
		}
	}
	return auths
}

// SetTime sets the decoding time of a transaction. This is used by tests to set
// arbitrary times and by persistent transaction pools when loading old txs from
// disk.
func (tx *Transaction) SetTime(t time.Time) {
	tx.time = t
}

// Time returns the time when the transaction was first seen on the network. It
// is a heuristic to prefer mining older txs vs new all other things equal.
func (tx *Transaction) Time() time.Time {
	return tx.time
}

// Hash returns the transaction hash.
func (tx *Transaction) Hash() common.Hash {
	if hash := tx.hash.Load(); hash != nil {
		return *hash
	}

	var h common.Hash
	if tx.Type() == LegacyTxType {
		h = rlpHash(tx.inner)
	} else if tx.Type() == ArbitrumLegacyTxType {
		h = tx.inner.(*ArbitrumLegacyTxData).HashOverride
	} else {
		h = prefixedRlpHash(tx.Type(), tx.inner)
	}

	if IsTargetBlock() {
		OLog2(fmt.Sprintf("transaction hash=%s", h.String()))
	}

	tx.hash.Store(&h)
	return h
}

// Size returns the true encoded storage size of the transaction, either by encoding
// and returning it, or returning a previously cached value.
func (tx *Transaction) Size() uint64 {
	if size := tx.size.Load(); size > 0 {
		return size
	}

	// Cache miss, encode and cache.
	// Note we rely on the assumption that all tx.inner values are RLP-encoded!
	c := writeCounter(0)
	rlp.Encode(&c, &tx.inner)
	size := uint64(c)

	// For blob transactions, add the size of the blob content and the outer list of the
	// tx + sidecar encoding.
	if sc := tx.BlobTxSidecar(); sc != nil {
		size += rlp.ListSize(sc.encodedSize())
	}

	// For typed transactions, the encoding also includes the leading type byte.
	if tx.Type() != LegacyTxType {
		size += 1
	}

	tx.size.Store(size)
	return size
}

// WithSignature returns a new transaction with the given signature.
// This signature needs to be in the [R || S || V] format where V is 0 or 1.
func (tx *Transaction) WithSignature(signer Signer, sig []byte) (*Transaction, error) {
	r, s, v, err := signer.SignatureValues(tx, sig)
	if err != nil {
		return nil, err
	}
	if r == nil || s == nil || v == nil {
		return nil, fmt.Errorf("%w: r: %s, s: %s, v: %s", ErrInvalidSig, r, s, v)
	}
	cpy := tx.inner.copy()
	cpy.setSignatureValues(signer.ChainID(), v, r, s)
	return &Transaction{inner: cpy, time: tx.time}, nil
}

// Transactions implements DerivableList for transactions.
type Transactions []*Transaction

// Len returns the length of s.
func (s Transactions) Len() int { return len(s) }

// EncodeIndex encodes the i'th transaction to w. Note that this does not check for errors
// because we assume that *Transaction will only ever contain valid txs that were either
// constructed by decoding or via public API in this package.
func (s Transactions) EncodeIndex(i int, w *bytes.Buffer) {
	tx := s[i]
	if tx.Type() == LegacyTxType {
		rlp.Encode(w, tx.inner)
	} else if tx.Type() == ArbitrumLegacyTxType {
		arbData := tx.inner.(*ArbitrumLegacyTxData)
		arbData.EncodeOnlyLegacyInto(w)
	} else {
		tx.encodeTyped(w)
	}
}

// TxDifference returns a new set of transactions that are present in a but not in b.
func TxDifference(a, b Transactions) Transactions {
	keep := make(Transactions, 0, len(a))

	remove := make(map[common.Hash]struct{}, b.Len())
	for _, tx := range b {
		remove[tx.Hash()] = struct{}{}
	}

	for _, tx := range a {
		if _, ok := remove[tx.Hash()]; !ok {
			keep = append(keep, tx)
		}
	}

	return keep
}

// HashDifference returns a new set of hashes that are present in a but not in b.
func HashDifference(a, b []common.Hash) []common.Hash {
	keep := make([]common.Hash, 0, len(a))

	remove := make(map[common.Hash]struct{})
	for _, hash := range b {
		remove[hash] = struct{}{}
	}

	for _, hash := range a {
		if _, ok := remove[hash]; !ok {
			keep = append(keep, hash)
		}
	}

	return keep
}

// TxByNonce implements the sort interface to allow sorting a list of transactions
// by their nonces. This is usually only useful for sorting transactions from a
// single account, otherwise a nonce comparison doesn't make much sense.
type TxByNonce Transactions

func (s TxByNonce) Len() int           { return len(s) }
func (s TxByNonce) Less(i, j int) bool { return s[i].Nonce() < s[j].Nonce() }
func (s TxByNonce) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// copyAddressPtr copies an address.
func copyAddressPtr(a *common.Address) *common.Address {
	if a == nil {
		return nil
	}
	cpy := *a
	return &cpy
}
