…Л
џѓ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
≠
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%ЌћL>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ю
MatrixSetDiagV3

input"T
diagonal"T
k
output"T"	
Ttype"Q
alignstring
RIGHT_LEFT:2
0
LEFT_RIGHT
RIGHT_LEFT	LEFT_LEFTRIGHT_RIGHT
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.6.32v2.6.2-194-g92a6bb065498ят

gat_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Щ

* 
shared_namegat_conv/kernel
x
#gat_conv/kernel/Read/ReadVariableOpReadVariableOpgat_conv/kernel*#
_output_shapes
:Щ

*
dtype0
Т
gat_conv/attn_kernel_selfVarHandleOp*
_output_shapes
: *
dtype0*
shape:

**
shared_namegat_conv/attn_kernel_self
Л
-gat_conv/attn_kernel_self/Read/ReadVariableOpReadVariableOpgat_conv/attn_kernel_self*"
_output_shapes
:

*
dtype0
Ф
gat_conv/attn_kernel_neighVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*+
shared_namegat_conv/attn_kernel_neigh
Н
.gat_conv/attn_kernel_neigh/Read/ReadVariableOpReadVariableOpgat_conv/attn_kernel_neigh*"
_output_shapes
:

*
dtype0
r
gat_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namegat_conv/bias
k
!gat_conv/bias/Read/ReadVariableOpReadVariableOpgat_conv/bias*
_output_shapes
:d*
dtype0
В
gat_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:d

*"
shared_namegat_conv_1/kernel
{
%gat_conv_1/kernel/Read/ReadVariableOpReadVariableOpgat_conv_1/kernel*"
_output_shapes
:d

*
dtype0
Ц
gat_conv_1/attn_kernel_selfVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*,
shared_namegat_conv_1/attn_kernel_self
П
/gat_conv_1/attn_kernel_self/Read/ReadVariableOpReadVariableOpgat_conv_1/attn_kernel_self*"
_output_shapes
:

*
dtype0
Ш
gat_conv_1/attn_kernel_neighVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*-
shared_namegat_conv_1/attn_kernel_neigh
С
0gat_conv_1/attn_kernel_neigh/Read/ReadVariableOpReadVariableOpgat_conv_1/attn_kernel_neigh*"
_output_shapes
:

*
dtype0
v
gat_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d* 
shared_namegat_conv_1/bias
o
#gat_conv_1/bias/Read/ReadVariableOpReadVariableOpgat_conv_1/bias*
_output_shapes
:d*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
Ч
RMSprop/gat_conv/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:Щ

*,
shared_nameRMSprop/gat_conv/kernel/rms
Р
/RMSprop/gat_conv/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gat_conv/kernel/rms*#
_output_shapes
:Щ

*
dtype0
™
%RMSprop/gat_conv/attn_kernel_self/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*6
shared_name'%RMSprop/gat_conv/attn_kernel_self/rms
£
9RMSprop/gat_conv/attn_kernel_self/rms/Read/ReadVariableOpReadVariableOp%RMSprop/gat_conv/attn_kernel_self/rms*"
_output_shapes
:

*
dtype0
ђ
&RMSprop/gat_conv/attn_kernel_neigh/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*7
shared_name(&RMSprop/gat_conv/attn_kernel_neigh/rms
•
:RMSprop/gat_conv/attn_kernel_neigh/rms/Read/ReadVariableOpReadVariableOp&RMSprop/gat_conv/attn_kernel_neigh/rms*"
_output_shapes
:

*
dtype0
К
RMSprop/gat_conv/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:d**
shared_nameRMSprop/gat_conv/bias/rms
Г
-RMSprop/gat_conv/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gat_conv/bias/rms*
_output_shapes
:d*
dtype0
Ъ
RMSprop/gat_conv_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:d

*.
shared_nameRMSprop/gat_conv_1/kernel/rms
У
1RMSprop/gat_conv_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gat_conv_1/kernel/rms*"
_output_shapes
:d

*
dtype0
Ѓ
'RMSprop/gat_conv_1/attn_kernel_self/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*8
shared_name)'RMSprop/gat_conv_1/attn_kernel_self/rms
І
;RMSprop/gat_conv_1/attn_kernel_self/rms/Read/ReadVariableOpReadVariableOp'RMSprop/gat_conv_1/attn_kernel_self/rms*"
_output_shapes
:

*
dtype0
∞
(RMSprop/gat_conv_1/attn_kernel_neigh/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*9
shared_name*(RMSprop/gat_conv_1/attn_kernel_neigh/rms
©
<RMSprop/gat_conv_1/attn_kernel_neigh/rms/Read/ReadVariableOpReadVariableOp(RMSprop/gat_conv_1/attn_kernel_neigh/rms*"
_output_shapes
:

*
dtype0
О
RMSprop/gat_conv_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*,
shared_nameRMSprop/gat_conv_1/bias/rms
З
/RMSprop/gat_conv_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gat_conv_1/bias/rms*
_output_shapes
:d*
dtype0
М
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*)
shared_nameRMSprop/dense/kernel/rms
Е
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes

:d*
dtype0
Д
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
Ц2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*—1
value«1Bƒ1 Bљ1
А
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
 
 
Ћ
kwargs_keys

kernel
attn_kernel_self
attn_kernel_neigh
attn_kernel_neighs
bias
dropout
trainable_variables
regularization_losses
	variables
	keras_api
Ћ
kwargs_keys

kernel
attn_kernel_self
attn_kernel_neigh
attn_kernel_neighs
bias
dropout
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
≠
&iter
	'decay
(learning_rate
)momentum
*rho	rmsb	rmsc	rmsd	rmse	rmsf	rmsg	rmsh	rmsi	 rmsj	!rmsk
F
0
1
2
3
4
5
6
7
 8
!9
 
F
0
1
2
3
4
5
6
7
 8
!9
≠
trainable_variables

+layers
regularization_losses
,metrics
		variables
-non_trainable_variables
.layer_regularization_losses
/layer_metrics
 
 
[Y
VARIABLE_VALUEgat_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEgat_conv/attn_kernel_self@layer_with_weights-0/attn_kernel_self/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEgat_conv/attn_kernel_neighAlayer_with_weights-0/attn_kernel_neigh/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEgat_conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
R
0trainable_variables
1regularization_losses
2	variables
3	keras_api

0
1
2
3
 

0
1
2
3
≠
trainable_variables

4layers
regularization_losses
5metrics
	variables
6non_trainable_variables
7layer_regularization_losses
8layer_metrics
 
][
VARIABLE_VALUEgat_conv_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEgat_conv_1/attn_kernel_self@layer_with_weights-1/attn_kernel_self/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEgat_conv_1/attn_kernel_neighAlayer_with_weights-1/attn_kernel_neigh/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgat_conv_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
R
9trainable_variables
:regularization_losses
;	variables
<	keras_api

0
1
2
3
 

0
1
2
3
≠
trainable_variables

=layers
regularization_losses
>metrics
	variables
?non_trainable_variables
@layer_regularization_losses
Alayer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
≠
"trainable_variables

Blayers
#regularization_losses
Cmetrics
$	variables
Dnon_trainable_variables
Elayer_regularization_losses
Flayer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4

G0
H1
I2
 
 
 
 
 
 
≠
0trainable_variables

Jlayers
1regularization_losses
Kmetrics
2	variables
Lnon_trainable_variables
Mlayer_regularization_losses
Nlayer_metrics

0
 
 
 
 
 
 
 
≠
9trainable_variables

Olayers
:regularization_losses
Pmetrics
;	variables
Qnon_trainable_variables
Rlayer_regularization_losses
Slayer_metrics

0
 
 
 
 
 
 
 
 
 
4
	Ttotal
	Ucount
V	variables
W	keras_api
D
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api
W
]
thresholds
^true_positives
_false_positives
`	variables
a	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

V	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

[	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

`	variables
ЖГ
VARIABLE_VALUERMSprop/gat_conv/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE%RMSprop/gat_conv/attn_kernel_self/rms^layer_with_weights-0/attn_kernel_self/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE&RMSprop/gat_conv/attn_kernel_neigh/rms_layer_with_weights-0/attn_kernel_neigh/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUERMSprop/gat_conv/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUERMSprop/gat_conv_1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE'RMSprop/gat_conv_1/attn_kernel_self/rms^layer_with_weights-1/attn_kernel_self/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE(RMSprop/gat_conv_1/attn_kernel_neigh/rms_layer_with_weights-1/attn_kernel_neigh/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUERMSprop/gat_conv_1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ж
serving_default_input_1Placeholder*-
_output_shapes
:€€€€€€€€€ФЩ*
dtype0*"
shape:€€€€€€€€€ФЩ
Ж
serving_default_input_2Placeholder*-
_output_shapes
:€€€€€€€€€ФФ*
dtype0*"
shape:€€€€€€€€€ФФ
≥
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2gat_conv/kernelgat_conv/attn_kernel_selfgat_conv/attn_kernel_neighgat_conv/biasgat_conv_1/kernelgat_conv_1/attn_kernel_selfgat_conv_1/attn_kernel_neighgat_conv_1/biasdense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_3887
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
–
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#gat_conv/kernel/Read/ReadVariableOp-gat_conv/attn_kernel_self/Read/ReadVariableOp.gat_conv/attn_kernel_neigh/Read/ReadVariableOp!gat_conv/bias/Read/ReadVariableOp%gat_conv_1/kernel/Read/ReadVariableOp/gat_conv_1/attn_kernel_self/Read/ReadVariableOp0gat_conv_1/attn_kernel_neigh/Read/ReadVariableOp#gat_conv_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp/RMSprop/gat_conv/kernel/rms/Read/ReadVariableOp9RMSprop/gat_conv/attn_kernel_self/rms/Read/ReadVariableOp:RMSprop/gat_conv/attn_kernel_neigh/rms/Read/ReadVariableOp-RMSprop/gat_conv/bias/rms/Read/ReadVariableOp1RMSprop/gat_conv_1/kernel/rms/Read/ReadVariableOp;RMSprop/gat_conv_1/attn_kernel_self/rms/Read/ReadVariableOp<RMSprop/gat_conv_1/attn_kernel_neigh/rms/Read/ReadVariableOp/RMSprop/gat_conv_1/bias/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_4684
я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegat_conv/kernelgat_conv/attn_kernel_selfgat_conv/attn_kernel_neighgat_conv/biasgat_conv_1/kernelgat_conv_1/attn_kernel_selfgat_conv_1/attn_kernel_neighgat_conv_1/biasdense/kernel
dense/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1true_positivesfalse_positivesRMSprop/gat_conv/kernel/rms%RMSprop/gat_conv/attn_kernel_self/rms&RMSprop/gat_conv/attn_kernel_neigh/rmsRMSprop/gat_conv/bias/rmsRMSprop/gat_conv_1/kernel/rms'RMSprop/gat_conv_1/attn_kernel_self/rms(RMSprop/gat_conv_1/attn_kernel_neigh/rmsRMSprop/gat_conv_1/bias/rmsRMSprop/dense/kernel/rmsRMSprop/dense/bias/rms*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_4787Мж
ъ	
л
'__inference_gat_conv_layer_call_fn_4363
inputs_0
inputs_1
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_gat_conv_layer_call_and_return_conditional_losses_33592
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ь
С
$__inference_dense_layer_call_fn_4567

inputs
unknown:d
	unknown_0:
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34712
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Фd: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Фd
 
_user_specified_nameinputs
±E
“
B__inference_gat_conv_layer_call_and_return_conditional_losses_4349
inputs_0
inputs_1<
%einsum_einsum_readvariableop_resource:Щ

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diagІ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02
einsum/Einsum/ReadVariableOpљ
einsum/EinsumEinsuminputs_0$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ІE
—
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_3430

inputs
inputs_1;
%einsum_einsum_readvariableop_resource:d

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diag¶
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02
einsum/Einsum/ReadVariableOpї
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€Фd:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Фd
 
_user_specified_nameinputs:UQ
-
_output_shapes
:€€€€€€€€€ФФ
 
_user_specified_nameinputs
–И
й
 __inference__traced_restore_4787
file_prefix7
 assignvariableop_gat_conv_kernel:Щ

B
,assignvariableop_1_gat_conv_attn_kernel_self:

C
-assignvariableop_2_gat_conv_attn_kernel_neigh:

.
 assignvariableop_3_gat_conv_bias:d:
$assignvariableop_4_gat_conv_1_kernel:d

D
.assignvariableop_5_gat_conv_1_attn_kernel_self:

E
/assignvariableop_6_gat_conv_1_attn_kernel_neigh:

0
"assignvariableop_7_gat_conv_1_bias:d1
assignvariableop_8_dense_kernel:d+
assignvariableop_9_dense_bias:*
 assignvariableop_10_rmsprop_iter:	 +
!assignvariableop_11_rmsprop_decay: 3
)assignvariableop_12_rmsprop_learning_rate: .
$assignvariableop_13_rmsprop_momentum: )
assignvariableop_14_rmsprop_rho: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: 0
"assignvariableop_19_true_positives:1
#assignvariableop_20_false_positives:F
/assignvariableop_21_rmsprop_gat_conv_kernel_rms:Щ

O
9assignvariableop_22_rmsprop_gat_conv_attn_kernel_self_rms:

P
:assignvariableop_23_rmsprop_gat_conv_attn_kernel_neigh_rms:

;
-assignvariableop_24_rmsprop_gat_conv_bias_rms:dG
1assignvariableop_25_rmsprop_gat_conv_1_kernel_rms:d

Q
;assignvariableop_26_rmsprop_gat_conv_1_attn_kernel_self_rms:

R
<assignvariableop_27_rmsprop_gat_conv_1_attn_kernel_neigh_rms:

=
/assignvariableop_28_rmsprop_gat_conv_1_bias_rms:d>
,assignvariableop_29_rmsprop_dense_kernel_rms:d8
*assignvariableop_30_rmsprop_dense_bias_rms:
identity_32ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ґ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*¬
valueЄBµ B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/attn_kernel_self/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/attn_kernel_neigh/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/attn_kernel_self/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/attn_kernel_neigh/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/attn_kernel_self/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-1/attn_kernel_neigh/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesќ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesќ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_gat_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_gat_conv_attn_kernel_selfIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2≤
AssignVariableOp_2AssignVariableOp-assignvariableop_2_gat_conv_attn_kernel_neighIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_gat_conv_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_gat_conv_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5≥
AssignVariableOp_5AssignVariableOp.assignvariableop_5_gat_conv_1_attn_kernel_selfIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6і
AssignVariableOp_6AssignVariableOp/assignvariableop_6_gat_conv_1_attn_kernel_neighIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7І
AssignVariableOp_7AssignVariableOp"assignvariableop_7_gat_conv_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ґ
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp assignvariableop_10_rmsprop_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11©
AssignVariableOp_11AssignVariableOp!assignvariableop_11_rmsprop_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12±
AssignVariableOp_12AssignVariableOp)assignvariableop_12_rmsprop_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ђ
AssignVariableOp_13AssignVariableOp$assignvariableop_13_rmsprop_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14І
AssignVariableOp_14AssignVariableOpassignvariableop_14_rmsprop_rhoIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15°
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19™
AssignVariableOp_19AssignVariableOp"assignvariableop_19_true_positivesIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ђ
AssignVariableOp_20AssignVariableOp#assignvariableop_20_false_positivesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ј
AssignVariableOp_21AssignVariableOp/assignvariableop_21_rmsprop_gat_conv_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ѕ
AssignVariableOp_22AssignVariableOp9assignvariableop_22_rmsprop_gat_conv_attn_kernel_self_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¬
AssignVariableOp_23AssignVariableOp:assignvariableop_23_rmsprop_gat_conv_attn_kernel_neigh_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24µ
AssignVariableOp_24AssignVariableOp-assignvariableop_24_rmsprop_gat_conv_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25є
AssignVariableOp_25AssignVariableOp1assignvariableop_25_rmsprop_gat_conv_1_kernel_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26√
AssignVariableOp_26AssignVariableOp;assignvariableop_26_rmsprop_gat_conv_1_attn_kernel_self_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27ƒ
AssignVariableOp_27AssignVariableOp<assignvariableop_27_rmsprop_gat_conv_1_attn_kernel_neigh_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ј
AssignVariableOp_28AssignVariableOp/assignvariableop_28_rmsprop_gat_conv_1_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29і
AssignVariableOp_29AssignVariableOp,assignvariableop_29_rmsprop_dense_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30≤
AssignVariableOp_30AssignVariableOp*assignvariableop_30_rmsprop_dense_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31f
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_32р
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ш
≠
?__inference_model_layer_call_and_return_conditional_losses_3853
input_1
input_2$
gat_conv_3829:Щ

#
gat_conv_3831:

#
gat_conv_3833:


gat_conv_3835:d%
gat_conv_1_3838:d

%
gat_conv_1_3840:

%
gat_conv_1_3842:


gat_conv_1_3844:d

dense_3847:d

dense_3849:
identityИҐdense/StatefulPartitionedCallҐ gat_conv/StatefulPartitionedCallҐ"gat_conv_1/StatefulPartitionedCallј
 gat_conv/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2gat_conv_3829gat_conv_3831gat_conv_3833gat_conv_3835*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_gat_conv_layer_call_and_return_conditional_losses_36802"
 gat_conv/StatefulPartitionedCallр
"gat_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gat_conv/StatefulPartitionedCall:output:0input_2gat_conv_1_3838gat_conv_1_3840gat_conv_1_3842gat_conv_1_3844*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_35902$
"gat_conv_1/StatefulPartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall+gat_conv_1/StatefulPartitionedCall:output:0
dense_3847
dense_3849*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34712
dense/StatefulPartitionedCallЖ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityґ
NoOpNoOp^dense/StatefulPartitionedCall!^gat_conv/StatefulPartitionedCall#^gat_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 gat_conv/StatefulPartitionedCall gat_conv/StatefulPartitionedCall2H
"gat_conv_1/StatefulPartitionedCall"gat_conv_1/StatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ФЩ
!
_user_specified_name	input_1:VR
-
_output_shapes
:€€€€€€€€€ФФ
!
_user_specified_name	input_2
ъ
Ш
$__inference_model_layer_call_fn_3797
input_1
input_2
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
	unknown_3:d


	unknown_4:


	unknown_5:


	unknown_6:d
	unknown_7:d
	unknown_8:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_37482
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ФЩ
!
_user_specified_name	input_1:VR
-
_output_shapes
:€€€€€€€€€ФФ
!
_user_specified_name	input_2
ъ
Ш
$__inference_model_layer_call_fn_3501
input_1
input_2
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
	unknown_3:d


	unknown_4:


	unknown_5:


	unknown_6:d
	unknown_7:d
	unknown_8:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_34782
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ФЩ
!
_user_specified_name	input_1:VR
-
_output_shapes
:€€€€€€€€€ФФ
!
_user_specified_name	input_2
Ў
Ц
"__inference_signature_wrapper_3887
input_1
input_2
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
	unknown_3:d


	unknown_4:


	unknown_5:


	unknown_6:d
	unknown_7:d
	unknown_8:
identityИҐStatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_32892
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ФЩ
!
_user_specified_name	input_1:VR
-
_output_shapes
:€€€€€€€€€ФФ
!
_user_specified_name	input_2
–!
ц
?__inference_dense_layer_call_and_return_conditional_losses_4558

inputs3
!tensordot_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2	
BiasAddf
SoftmaxSoftmaxBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2	
Softmaxq
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

IdentityВ
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Фd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Фd
 
_user_specified_nameinputs
А
Ъ
$__inference_model_layer_call_fn_4201
inputs_0
inputs_1
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
	unknown_3:d


	unknown_4:


	unknown_5:


	unknown_6:d
	unknown_7:d
	unknown_8:
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_34782
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ч
≠
?__inference_model_layer_call_and_return_conditional_losses_3478

inputs
inputs_1$
gat_conv_3360:Щ

#
gat_conv_3362:

#
gat_conv_3364:


gat_conv_3366:d%
gat_conv_1_3431:d

%
gat_conv_1_3433:

%
gat_conv_1_3435:


gat_conv_1_3437:d

dense_3472:d

dense_3474:
identityИҐdense/StatefulPartitionedCallҐ gat_conv/StatefulPartitionedCallҐ"gat_conv_1/StatefulPartitionedCallј
 gat_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1gat_conv_3360gat_conv_3362gat_conv_3364gat_conv_3366*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_gat_conv_layer_call_and_return_conditional_losses_33592"
 gat_conv/StatefulPartitionedCallс
"gat_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gat_conv/StatefulPartitionedCall:output:0inputs_1gat_conv_1_3431gat_conv_1_3433gat_conv_1_3435gat_conv_1_3437*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_34302$
"gat_conv_1/StatefulPartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall+gat_conv_1/StatefulPartitionedCall:output:0
dense_3472
dense_3474*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34712
dense/StatefulPartitionedCallЖ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityґ
NoOpNoOp^dense/StatefulPartitionedCall!^gat_conv/StatefulPartitionedCall#^gat_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 gat_conv/StatefulPartitionedCall gat_conv/StatefulPartitionedCall2H
"gat_conv_1/StatefulPartitionedCall"gat_conv_1/StatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ФЩ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:€€€€€€€€€ФФ
 
_user_specified_nameinputs
ОЏ
’	
__inference__wrapped_model_3289
input_1
input_2K
4model_gat_conv_einsum_einsum_readvariableop_resource:Щ

L
6model_gat_conv_einsum_1_einsum_readvariableop_resource:

L
6model_gat_conv_einsum_2_einsum_readvariableop_resource:

:
,model_gat_conv_add_3_readvariableop_resource:dL
6model_gat_conv_1_einsum_einsum_readvariableop_resource:d

N
8model_gat_conv_1_einsum_1_einsum_readvariableop_resource:

N
8model_gat_conv_1_einsum_2_einsum_readvariableop_resource:

<
.model_gat_conv_1_add_3_readvariableop_resource:d?
-model_dense_tensordot_readvariableop_resource:d9
+model_dense_biasadd_readvariableop_resource:
identityИҐ"model/dense/BiasAdd/ReadVariableOpҐ$model/dense/Tensordot/ReadVariableOpҐ#model/gat_conv/add_3/ReadVariableOpҐ+model/gat_conv/einsum/Einsum/ReadVariableOpҐ-model/gat_conv/einsum_1/Einsum/ReadVariableOpҐ-model/gat_conv/einsum_2/Einsum/ReadVariableOpҐ%model/gat_conv_1/add_3/ReadVariableOpҐ-model/gat_conv_1/einsum/Einsum/ReadVariableOpҐ/model/gat_conv_1/einsum_1/Einsum/ReadVariableOpҐ/model/gat_conv_1/einsum_2/Einsum/ReadVariableOpc
model/gat_conv/ShapeShapeinput_2*
T0*
_output_shapes
:2
model/gat_conv/ShapeТ
"model/gat_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"model/gat_conv/strided_slice/stackЯ
$model/gat_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2&
$model/gat_conv/strided_slice/stack_1Ц
$model/gat_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$model/gat_conv/strided_slice/stack_2Ї
model/gat_conv/strided_sliceStridedSlicemodel/gat_conv/Shape:output:0+model/gat_conv/strided_slice/stack:output:0-model/gat_conv/strided_slice/stack_1:output:0-model/gat_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
model/gat_conv/strided_sliceЧ
$model/gat_conv/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2&
$model/gat_conv/Sum/reduction_indicesЪ
model/gat_conv/SumSuminput_2-model/gat_conv/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
model/gat_conv/SumЫ
&model/gat_conv/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2(
&model/gat_conv/Sum_1/reduction_indices†
model/gat_conv/Sum_1Suminput_2/model/gat_conv/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
model/gat_conv/Sum_1†
model/gat_conv/addAddV2model/gat_conv/Sum_1:output:0model/gat_conv/Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
model/gat_conv/addx
model/gat_conv/set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gat_conv/set_diag/k¬
model/gat_conv/set_diagMatrixSetDiagV3input_2model/gat_conv/add:z:0"model/gat_conv/set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
model/gat_conv/set_diag‘
+model/gat_conv/einsum/Einsum/ReadVariableOpReadVariableOp4model_gat_conv_einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02-
+model/gat_conv/einsum/Einsum/ReadVariableOpй
model/gat_conv/einsum/EinsumEinsuminput_13model/gat_conv/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
model/gat_conv/einsum/Einsumў
-model/gat_conv/einsum_1/Einsum/ReadVariableOpReadVariableOp6model_gat_conv_einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02/
-model/gat_conv/einsum_1/Einsum/ReadVariableOpО
model/gat_conv/einsum_1/EinsumEinsum%model/gat_conv/einsum/Einsum:output:05model/gat_conv/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2 
model/gat_conv/einsum_1/Einsumў
-model/gat_conv/einsum_2/Einsum/ReadVariableOpReadVariableOp6model_gat_conv_einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02/
-model/gat_conv/einsum_2/Einsum/ReadVariableOpО
model/gat_conv/einsum_2/EinsumEinsum%model/gat_conv/einsum/Einsum:output:05model/gat_conv/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2 
model/gat_conv/einsum_2/Einsum’
model/gat_conv/einsum_3/EinsumEinsum'model/gat_conv/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2 
model/gat_conv/einsum_3/Einsum√
model/gat_conv/add_1AddV2'model/gat_conv/einsum_1/Einsum:output:0'model/gat_conv/einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv/add_1О
model/gat_conv/LeakyRelu	LeakyRelumodel/gat_conv/add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv/LeakyRelui
model/gat_conv/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gat_conv/x 
model/gat_conv/EqualEqual model/gat_conv/set_diag:output:0model/gat_conv/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
model/gat_conv/Equal{
model/gat_conv/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2
model/gat_conv/SelectV2/t{
model/gat_conv/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gat_conv/SelectV2/eЎ
model/gat_conv/SelectV2SelectV2model/gat_conv/Equal:z:0"model/gat_conv/SelectV2/t:output:0"model/gat_conv/SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
model/gat_conv/SelectV2°
$model/gat_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2&
$model/gat_conv/strided_slice_1/stack•
&model/gat_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2(
&model/gat_conv/strided_slice_1/stack_1•
&model/gat_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2(
&model/gat_conv/strided_slice_1/stack_2Ш
model/gat_conv/strided_slice_1StridedSlice model/gat_conv/SelectV2:output:0-model/gat_conv/strided_slice_1/stack:output:0/model/gat_conv/strided_slice_1/stack_1:output:0/model/gat_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2 
model/gat_conv/strided_slice_1¬
model/gat_conv/add_2AddV2&model/gat_conv/LeakyRelu:activations:0'model/gat_conv/strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv/add_2С
model/gat_conv/SoftmaxSoftmaxmodel/gat_conv/add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv/SoftmaxС
$model/gat_conv/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$model/gat_conv/dropout/dropout/Const№
"model/gat_conv/dropout/dropout/MulMul model/gat_conv/Softmax:softmax:0-model/gat_conv/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2$
"model/gat_conv/dropout/dropout/MulЬ
$model/gat_conv/dropout/dropout/ShapeShape model/gat_conv/Softmax:softmax:0*
T0*
_output_shapes
:2&
$model/gat_conv/dropout/dropout/ShapeГ
;model/gat_conv/dropout/dropout/random_uniform/RandomUniformRandomUniform-model/gat_conv/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02=
;model/gat_conv/dropout/dropout/random_uniform/RandomUniform£
-model/gat_conv/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2/
-model/gat_conv/dropout/dropout/GreaterEqual/y§
+model/gat_conv/dropout/dropout/GreaterEqualGreaterEqualDmodel/gat_conv/dropout/dropout/random_uniform/RandomUniform:output:06model/gat_conv/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2-
+model/gat_conv/dropout/dropout/GreaterEqualќ
#model/gat_conv/dropout/dropout/CastCast/model/gat_conv/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2%
#model/gat_conv/dropout/dropout/Castа
$model/gat_conv/dropout/dropout/Mul_1Mul&model/gat_conv/dropout/dropout/Mul:z:0'model/gat_conv/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2&
$model/gat_conv/dropout/dropout/Mul_1Д
model/gat_conv/einsum_4/EinsumEinsum(model/gat_conv/dropout/dropout/Mul_1:z:0%model/gat_conv/einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2 
model/gat_conv/einsum_4/EinsumЗ
model/gat_conv/Shape_1Shape'model/gat_conv/einsum_4/Einsum:output:0*
T0*
_output_shapes
:2
model/gat_conv/Shape_1Ц
$model/gat_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model/gat_conv/strided_slice_2/stack£
&model/gat_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2(
&model/gat_conv/strided_slice_2/stack_1Ъ
&model/gat_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model/gat_conv/strided_slice_2/stack_2∆
model/gat_conv/strided_slice_2StridedSlicemodel/gat_conv/Shape_1:output:0-model/gat_conv/strided_slice_2/stack:output:0/model/gat_conv/strided_slice_2/stack_1:output:0/model/gat_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
model/gat_conv/strided_slice_2К
model/gat_conv/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2 
model/gat_conv/concat/values_1z
model/gat_conv/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gat_conv/concat/axisя
model/gat_conv/concatConcatV2'model/gat_conv/strided_slice_2:output:0'model/gat_conv/concat/values_1:output:0#model/gat_conv/concat/axis:output:0*
N*
T0*
_output_shapes
:2
model/gat_conv/concatї
model/gat_conv/ReshapeReshape'model/gat_conv/einsum_4/Einsum:output:0model/gat_conv/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
model/gat_conv/Reshape≥
#model/gat_conv/add_3/ReadVariableOpReadVariableOp,model_gat_conv_add_3_readvariableop_resource*
_output_shapes
:d*
dtype02%
#model/gat_conv/add_3/ReadVariableOpЇ
model/gat_conv/add_3AddV2model/gat_conv/Reshape:output:0+model/gat_conv/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
model/gat_conv/add_3g
model/gat_conv_1/ShapeShapeinput_2*
T0*
_output_shapes
:2
model/gat_conv_1/ShapeЦ
$model/gat_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$model/gat_conv_1/strided_slice/stack£
&model/gat_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2(
&model/gat_conv_1/strided_slice/stack_1Ъ
&model/gat_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&model/gat_conv_1/strided_slice/stack_2∆
model/gat_conv_1/strided_sliceStridedSlicemodel/gat_conv_1/Shape:output:0-model/gat_conv_1/strided_slice/stack:output:0/model/gat_conv_1/strided_slice/stack_1:output:0/model/gat_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
model/gat_conv_1/strided_sliceЫ
&model/gat_conv_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2(
&model/gat_conv_1/Sum/reduction_indices†
model/gat_conv_1/SumSuminput_2/model/gat_conv_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
model/gat_conv_1/SumЯ
(model/gat_conv_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2*
(model/gat_conv_1/Sum_1/reduction_indices¶
model/gat_conv_1/Sum_1Suminput_21model/gat_conv_1/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
model/gat_conv_1/Sum_1®
model/gat_conv_1/addAddV2model/gat_conv_1/Sum_1:output:0model/gat_conv_1/Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
model/gat_conv_1/add|
model/gat_conv_1/set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gat_conv_1/set_diag/k 
model/gat_conv_1/set_diagMatrixSetDiagV3input_2model/gat_conv_1/add:z:0$model/gat_conv_1/set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
model/gat_conv_1/set_diagў
-model/gat_conv_1/einsum/Einsum/ReadVariableOpReadVariableOp6model_gat_conv_1_einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02/
-model/gat_conv_1/einsum/Einsum/ReadVariableOpА
model/gat_conv_1/einsum/EinsumEinsummodel/gat_conv/add_3:z:05model/gat_conv_1/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2 
model/gat_conv_1/einsum/Einsumя
/model/gat_conv_1/einsum_1/Einsum/ReadVariableOpReadVariableOp8model_gat_conv_1_einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype021
/model/gat_conv_1/einsum_1/Einsum/ReadVariableOpЦ
 model/gat_conv_1/einsum_1/EinsumEinsum'model/gat_conv_1/einsum/Einsum:output:07model/gat_conv_1/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2"
 model/gat_conv_1/einsum_1/Einsumя
/model/gat_conv_1/einsum_2/Einsum/ReadVariableOpReadVariableOp8model_gat_conv_1_einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype021
/model/gat_conv_1/einsum_2/Einsum/ReadVariableOpЦ
 model/gat_conv_1/einsum_2/EinsumEinsum'model/gat_conv_1/einsum/Einsum:output:07model/gat_conv_1/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2"
 model/gat_conv_1/einsum_2/Einsumџ
 model/gat_conv_1/einsum_3/EinsumEinsum)model/gat_conv_1/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2"
 model/gat_conv_1/einsum_3/EinsumЋ
model/gat_conv_1/add_1AddV2)model/gat_conv_1/einsum_1/Einsum:output:0)model/gat_conv_1/einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv_1/add_1Ф
model/gat_conv_1/LeakyRelu	LeakyRelumodel/gat_conv_1/add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv_1/LeakyRelum
model/gat_conv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gat_conv_1/x“
model/gat_conv_1/EqualEqual"model/gat_conv_1/set_diag:output:0model/gat_conv_1/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
model/gat_conv_1/Equal
model/gat_conv_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2
model/gat_conv_1/SelectV2/t
model/gat_conv_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gat_conv_1/SelectV2/eв
model/gat_conv_1/SelectV2SelectV2model/gat_conv_1/Equal:z:0$model/gat_conv_1/SelectV2/t:output:0$model/gat_conv_1/SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
model/gat_conv_1/SelectV2•
&model/gat_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2(
&model/gat_conv_1/strided_slice_1/stack©
(model/gat_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2*
(model/gat_conv_1/strided_slice_1/stack_1©
(model/gat_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2*
(model/gat_conv_1/strided_slice_1/stack_2§
 model/gat_conv_1/strided_slice_1StridedSlice"model/gat_conv_1/SelectV2:output:0/model/gat_conv_1/strided_slice_1/stack:output:01model/gat_conv_1/strided_slice_1/stack_1:output:01model/gat_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2"
 model/gat_conv_1/strided_slice_1 
model/gat_conv_1/add_2AddV2(model/gat_conv_1/LeakyRelu:activations:0)model/gat_conv_1/strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv_1/add_2Ч
model/gat_conv_1/SoftmaxSoftmaxmodel/gat_conv_1/add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
model/gat_conv_1/SoftmaxХ
&model/gat_conv_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&model/gat_conv_1/dropout/dropout/Constд
$model/gat_conv_1/dropout/dropout/MulMul"model/gat_conv_1/Softmax:softmax:0/model/gat_conv_1/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2&
$model/gat_conv_1/dropout/dropout/MulҐ
&model/gat_conv_1/dropout/dropout/ShapeShape"model/gat_conv_1/Softmax:softmax:0*
T0*
_output_shapes
:2(
&model/gat_conv_1/dropout/dropout/ShapeЙ
=model/gat_conv_1/dropout/dropout/random_uniform/RandomUniformRandomUniform/model/gat_conv_1/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02?
=model/gat_conv_1/dropout/dropout/random_uniform/RandomUniformІ
/model/gat_conv_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?21
/model/gat_conv_1/dropout/dropout/GreaterEqual/yђ
-model/gat_conv_1/dropout/dropout/GreaterEqualGreaterEqualFmodel/gat_conv_1/dropout/dropout/random_uniform/RandomUniform:output:08model/gat_conv_1/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2/
-model/gat_conv_1/dropout/dropout/GreaterEqual‘
%model/gat_conv_1/dropout/dropout/CastCast1model/gat_conv_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2'
%model/gat_conv_1/dropout/dropout/Castи
&model/gat_conv_1/dropout/dropout/Mul_1Mul(model/gat_conv_1/dropout/dropout/Mul:z:0)model/gat_conv_1/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2(
&model/gat_conv_1/dropout/dropout/Mul_1М
 model/gat_conv_1/einsum_4/EinsumEinsum*model/gat_conv_1/dropout/dropout/Mul_1:z:0'model/gat_conv_1/einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2"
 model/gat_conv_1/einsum_4/EinsumН
model/gat_conv_1/Shape_1Shape)model/gat_conv_1/einsum_4/Einsum:output:0*
T0*
_output_shapes
:2
model/gat_conv_1/Shape_1Ъ
&model/gat_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model/gat_conv_1/strided_slice_2/stackІ
(model/gat_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2*
(model/gat_conv_1/strided_slice_2/stack_1Ю
(model/gat_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/gat_conv_1/strided_slice_2/stack_2“
 model/gat_conv_1/strided_slice_2StridedSlice!model/gat_conv_1/Shape_1:output:0/model/gat_conv_1/strided_slice_2/stack:output:01model/gat_conv_1/strided_slice_2/stack_1:output:01model/gat_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 model/gat_conv_1/strided_slice_2О
 model/gat_conv_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2"
 model/gat_conv_1/concat/values_1~
model/gat_conv_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gat_conv_1/concat/axisй
model/gat_conv_1/concatConcatV2)model/gat_conv_1/strided_slice_2:output:0)model/gat_conv_1/concat/values_1:output:0%model/gat_conv_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
model/gat_conv_1/concat√
model/gat_conv_1/ReshapeReshape)model/gat_conv_1/einsum_4/Einsum:output:0 model/gat_conv_1/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
model/gat_conv_1/Reshapeє
%model/gat_conv_1/add_3/ReadVariableOpReadVariableOp.model_gat_conv_1_add_3_readvariableop_resource*
_output_shapes
:d*
dtype02'
%model/gat_conv_1/add_3/ReadVariableOp¬
model/gat_conv_1/add_3AddV2!model/gat_conv_1/Reshape:output:0-model/gat_conv_1/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
model/gat_conv_1/add_3Ї
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype02&
$model/dense/Tensordot/ReadVariableOpВ
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
model/dense/Tensordot/axesЙ
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
model/dense/Tensordot/freeД
model/dense/Tensordot/ShapeShapemodel/gat_conv_1/add_3:z:0*
T0*
_output_shapes
:2
model/dense/Tensordot/ShapeМ
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/dense/Tensordot/GatherV2/axisН
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
model/dense/Tensordot/GatherV2Р
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model/dense/Tensordot/GatherV2_1/axisУ
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 model/dense/Tensordot/GatherV2_1Д
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
model/dense/Tensordot/Const∞
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
model/dense/Tensordot/ProdИ
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
model/dense/Tensordot/Const_1Є
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
model/dense/Tensordot/Prod_1И
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model/dense/Tensordot/concat/axisм
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
model/dense/Tensordot/concatЉ
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
model/dense/Tensordot/stack…
model/dense/Tensordot/transpose	Transposemodel/gat_conv_1/add_3:z:0%model/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2!
model/dense/Tensordot/transposeѕ
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
model/dense/Tensordot/Reshapeќ
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
model/dense/Tensordot/MatMulИ
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
model/dense/Tensordot/Const_2М
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/dense/Tensordot/concat_1/axisщ
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
model/dense/Tensordot/concat_1Ѕ
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
model/dense/Tensordot∞
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOpЄ
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
model/dense/BiasAddК
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
model/dense/Softmax}
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

IdentityК
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp$^model/gat_conv/add_3/ReadVariableOp,^model/gat_conv/einsum/Einsum/ReadVariableOp.^model/gat_conv/einsum_1/Einsum/ReadVariableOp.^model/gat_conv/einsum_2/Einsum/ReadVariableOp&^model/gat_conv_1/add_3/ReadVariableOp.^model/gat_conv_1/einsum/Einsum/ReadVariableOp0^model/gat_conv_1/einsum_1/Einsum/ReadVariableOp0^model/gat_conv_1/einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2J
#model/gat_conv/add_3/ReadVariableOp#model/gat_conv/add_3/ReadVariableOp2Z
+model/gat_conv/einsum/Einsum/ReadVariableOp+model/gat_conv/einsum/Einsum/ReadVariableOp2^
-model/gat_conv/einsum_1/Einsum/ReadVariableOp-model/gat_conv/einsum_1/Einsum/ReadVariableOp2^
-model/gat_conv/einsum_2/Einsum/ReadVariableOp-model/gat_conv/einsum_2/Einsum/ReadVariableOp2N
%model/gat_conv_1/add_3/ReadVariableOp%model/gat_conv_1/add_3/ReadVariableOp2^
-model/gat_conv_1/einsum/Einsum/ReadVariableOp-model/gat_conv_1/einsum/Einsum/ReadVariableOp2b
/model/gat_conv_1/einsum_1/Einsum/ReadVariableOp/model/gat_conv_1/einsum_1/Einsum/ReadVariableOp2b
/model/gat_conv_1/einsum_2/Einsum/ReadVariableOp/model/gat_conv_1/einsum_2/Einsum/ReadVariableOp:V R
-
_output_shapes
:€€€€€€€€€ФЩ
!
_user_specified_name	input_1:VR
-
_output_shapes
:€€€€€€€€€ФФ
!
_user_specified_name	input_2
©E
–
B__inference_gat_conv_layer_call_and_return_conditional_losses_3680

inputs
inputs_1<
%einsum_einsum_readvariableop_resource:Щ

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diagІ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02
einsum/Einsum/ReadVariableOpї
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€ФЩ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:€€€€€€€€€ФФ
 
_user_specified_nameinputs
–!
ц
?__inference_dense_layer_call_and_return_conditional_losses_3471

inputs3
!tensordot_readvariableop_resource:d-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:d*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2	
BiasAddf
SoftmaxSoftmaxBiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2	
Softmaxq
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

IdentityВ
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Фd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Фd
 
_user_specified_nameinputs
ъ	
л
'__inference_gat_conv_layer_call_fn_4377
inputs_0
inputs_1
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
identityИҐStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_gat_conv_layer_call_and_return_conditional_losses_36802
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ы	
м
)__inference_gat_conv_1_layer_call_fn_4513
inputs_0
inputs_1
unknown:d


	unknown_0:


	unknown_1:


	unknown_2:d
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_34302
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€Фd:€€€€€€€€€ФФ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:€€€€€€€€€Фd
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ш
≠
?__inference_model_layer_call_and_return_conditional_losses_3825
input_1
input_2$
gat_conv_3801:Щ

#
gat_conv_3803:

#
gat_conv_3805:


gat_conv_3807:d%
gat_conv_1_3810:d

%
gat_conv_1_3812:

%
gat_conv_1_3814:


gat_conv_1_3816:d

dense_3819:d

dense_3821:
identityИҐdense/StatefulPartitionedCallҐ gat_conv/StatefulPartitionedCallҐ"gat_conv_1/StatefulPartitionedCallј
 gat_conv/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2gat_conv_3801gat_conv_3803gat_conv_3805gat_conv_3807*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_gat_conv_layer_call_and_return_conditional_losses_33592"
 gat_conv/StatefulPartitionedCallр
"gat_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gat_conv/StatefulPartitionedCall:output:0input_2gat_conv_1_3810gat_conv_1_3812gat_conv_1_3814gat_conv_1_3816*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_34302$
"gat_conv_1/StatefulPartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall+gat_conv_1/StatefulPartitionedCall:output:0
dense_3819
dense_3821*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34712
dense/StatefulPartitionedCallЖ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityґ
NoOpNoOp^dense/StatefulPartitionedCall!^gat_conv/StatefulPartitionedCall#^gat_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 gat_conv/StatefulPartitionedCall gat_conv/StatefulPartitionedCall2H
"gat_conv_1/StatefulPartitionedCall"gat_conv_1/StatefulPartitionedCall:V R
-
_output_shapes
:€€€€€€€€€ФЩ
!
_user_specified_name	input_1:VR
-
_output_shapes
:€€€€€€€€€ФФ
!
_user_specified_name	input_2
ѓE
”
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_4499
inputs_0
inputs_1;
%einsum_einsum_readvariableop_resource:d

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diag¶
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02
einsum/Einsum/ReadVariableOpљ
einsum/EinsumEinsuminputs_0$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€Фd:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:V R
,
_output_shapes
:€€€€€€€€€Фd
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ы	
м
)__inference_gat_conv_1_layer_call_fn_4527
inputs_0
inputs_1
unknown:d


	unknown_0:


	unknown_1:


	unknown_2:d
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_35902
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€Фd:€€€€€€€€€ФФ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:€€€€€€€€€Фd
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
яG
„
__inference__traced_save_4684
file_prefix.
*savev2_gat_conv_kernel_read_readvariableop8
4savev2_gat_conv_attn_kernel_self_read_readvariableop9
5savev2_gat_conv_attn_kernel_neigh_read_readvariableop,
(savev2_gat_conv_bias_read_readvariableop0
,savev2_gat_conv_1_kernel_read_readvariableop:
6savev2_gat_conv_1_attn_kernel_self_read_readvariableop;
7savev2_gat_conv_1_attn_kernel_neigh_read_readvariableop.
*savev2_gat_conv_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop:
6savev2_rmsprop_gat_conv_kernel_rms_read_readvariableopD
@savev2_rmsprop_gat_conv_attn_kernel_self_rms_read_readvariableopE
Asavev2_rmsprop_gat_conv_attn_kernel_neigh_rms_read_readvariableop8
4savev2_rmsprop_gat_conv_bias_rms_read_readvariableop<
8savev2_rmsprop_gat_conv_1_kernel_rms_read_readvariableopF
Bsavev2_rmsprop_gat_conv_1_attn_kernel_self_rms_read_readvariableopG
Csavev2_rmsprop_gat_conv_1_attn_kernel_neigh_rms_read_readvariableop:
6savev2_rmsprop_gat_conv_1_bias_rms_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename∞
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*¬
valueЄBµ B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/attn_kernel_self/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-0/attn_kernel_neigh/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/attn_kernel_self/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-1/attn_kernel_neigh/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-0/attn_kernel_self/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-0/attn_kernel_neigh/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/attn_kernel_self/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-1/attn_kernel_neigh/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names»
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_gat_conv_kernel_read_readvariableop4savev2_gat_conv_attn_kernel_self_read_readvariableop5savev2_gat_conv_attn_kernel_neigh_read_readvariableop(savev2_gat_conv_bias_read_readvariableop,savev2_gat_conv_1_kernel_read_readvariableop6savev2_gat_conv_1_attn_kernel_self_read_readvariableop7savev2_gat_conv_1_attn_kernel_neigh_read_readvariableop*savev2_gat_conv_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop6savev2_rmsprop_gat_conv_kernel_rms_read_readvariableop@savev2_rmsprop_gat_conv_attn_kernel_self_rms_read_readvariableopAsavev2_rmsprop_gat_conv_attn_kernel_neigh_rms_read_readvariableop4savev2_rmsprop_gat_conv_bias_rms_read_readvariableop8savev2_rmsprop_gat_conv_1_kernel_rms_read_readvariableopBsavev2_rmsprop_gat_conv_1_attn_kernel_self_rms_read_readvariableopCsavev2_rmsprop_gat_conv_1_attn_kernel_neigh_rms_read_readvariableop6savev2_rmsprop_gat_conv_1_bias_rms_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Щ
_input_shapesЗ
Д: :Щ

:

:

:d:d

:

:

:d:d:: : : : : : : : : :::Щ

:

:

:d:d

:

:

:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:Щ

:($
"
_output_shapes
:

:($
"
_output_shapes
:

: 

_output_shapes
:d:($
"
_output_shapes
:d

:($
"
_output_shapes
:

:($
"
_output_shapes
:

: 

_output_shapes
:d:$	 

_output_shapes

:d: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::)%
#
_output_shapes
:Щ

:($
"
_output_shapes
:

:($
"
_output_shapes
:

: 

_output_shapes
:d:($
"
_output_shapes
:d

:($
"
_output_shapes
:

:($
"
_output_shapes
:

: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
:: 

_output_shapes
: 
±E
“
B__inference_gat_conv_layer_call_and_return_conditional_losses_4288
inputs_0
inputs_1<
%einsum_einsum_readvariableop_resource:Щ

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diagІ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02
einsum/Einsum/ReadVariableOpљ
einsum/EinsumEinsuminputs_0$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
©E
–
B__inference_gat_conv_layer_call_and_return_conditional_losses_3359

inputs
inputs_1<
%einsum_einsum_readvariableop_resource:Щ

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diagІ
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02
einsum/Einsum/ReadVariableOpї
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€ФЩ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:€€€€€€€€€ФФ
 
_user_specified_nameinputs
А
Ъ
$__inference_model_layer_call_fn_4227
inputs_0
inputs_1
unknown:Щ


	unknown_0:


	unknown_1:


	unknown_2:d
	unknown_3:d


	unknown_4:


	unknown_5:


	unknown_6:d
	unknown_7:d
	unknown_8:
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_37482
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ч
≠
?__inference_model_layer_call_and_return_conditional_losses_3748

inputs
inputs_1$
gat_conv_3724:Щ

#
gat_conv_3726:

#
gat_conv_3728:


gat_conv_3730:d%
gat_conv_1_3733:d

%
gat_conv_1_3735:

%
gat_conv_1_3737:


gat_conv_1_3739:d

dense_3742:d

dense_3744:
identityИҐdense/StatefulPartitionedCallҐ gat_conv/StatefulPartitionedCallҐ"gat_conv_1/StatefulPartitionedCallј
 gat_conv/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1gat_conv_3724gat_conv_3726gat_conv_3728gat_conv_3730*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_gat_conv_layer_call_and_return_conditional_losses_36802"
 gat_conv/StatefulPartitionedCallс
"gat_conv_1/StatefulPartitionedCallStatefulPartitionedCall)gat_conv/StatefulPartitionedCall:output:0inputs_1gat_conv_1_3733gat_conv_1_3735gat_conv_1_3737gat_conv_1_3739*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Фd*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_35902$
"gat_conv_1/StatefulPartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall+gat_conv_1/StatefulPartitionedCall:output:0
dense_3742
dense_3744*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Ф*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_34712
dense/StatefulPartitionedCallЖ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityґ
NoOpNoOp^dense/StatefulPartitionedCall!^gat_conv/StatefulPartitionedCall#^gat_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2D
 gat_conv/StatefulPartitionedCall gat_conv/StatefulPartitionedCall2H
"gat_conv_1/StatefulPartitionedCall"gat_conv_1/StatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€ФЩ
 
_user_specified_nameinputs:UQ
-
_output_shapes
:€€€€€€€€€ФФ
 
_user_specified_nameinputs
ѓE
”
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_4438
inputs_0
inputs_1;
%einsum_einsum_readvariableop_resource:d

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diag¶
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02
einsum/Einsum/ReadVariableOpљ
einsum/EinsumEinsuminputs_0$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€Фd:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:V R
,
_output_shapes
:€€€€€€€€€Фd
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
тƒ
€
?__inference_model_layer_call_and_return_conditional_losses_4031
inputs_0
inputs_1E
.gat_conv_einsum_einsum_readvariableop_resource:Щ

F
0gat_conv_einsum_1_einsum_readvariableop_resource:

F
0gat_conv_einsum_2_einsum_readvariableop_resource:

4
&gat_conv_add_3_readvariableop_resource:dF
0gat_conv_1_einsum_einsum_readvariableop_resource:d

H
2gat_conv_1_einsum_1_einsum_readvariableop_resource:

H
2gat_conv_1_einsum_2_einsum_readvariableop_resource:

6
(gat_conv_1_add_3_readvariableop_resource:d9
'dense_tensordot_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐgat_conv/add_3/ReadVariableOpҐ%gat_conv/einsum/Einsum/ReadVariableOpҐ'gat_conv/einsum_1/Einsum/ReadVariableOpҐ'gat_conv/einsum_2/Einsum/ReadVariableOpҐgat_conv_1/add_3/ReadVariableOpҐ'gat_conv_1/einsum/Einsum/ReadVariableOpҐ)gat_conv_1/einsum_1/Einsum/ReadVariableOpҐ)gat_conv_1/einsum_2/Einsum/ReadVariableOpX
gat_conv/ShapeShapeinputs_1*
T0*
_output_shapes
:2
gat_conv/ShapeЖ
gat_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gat_conv/strided_slice/stackУ
gat_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2 
gat_conv/strided_slice/stack_1К
gat_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gat_conv/strided_slice/stack_2Ц
gat_conv/strided_sliceStridedSlicegat_conv/Shape:output:0%gat_conv/strided_slice/stack:output:0'gat_conv/strided_slice/stack_1:output:0'gat_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv/strided_sliceЛ
gat_conv/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2 
gat_conv/Sum/reduction_indicesЙ
gat_conv/SumSuminputs_1'gat_conv/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv/SumП
 gat_conv/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2"
 gat_conv/Sum_1/reduction_indicesП
gat_conv/Sum_1Suminputs_1)gat_conv/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv/Sum_1И
gat_conv/addAddV2gat_conv/Sum_1:output:0gat_conv/Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv/addl
gat_conv/set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv/set_diag/kЂ
gat_conv/set_diagMatrixSetDiagV3inputs_1gat_conv/add:z:0gat_conv/set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv/set_diag¬
%gat_conv/einsum/Einsum/ReadVariableOpReadVariableOp.gat_conv_einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02'
%gat_conv/einsum/Einsum/ReadVariableOpЎ
gat_conv/einsum/EinsumEinsuminputs_0-gat_conv/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
gat_conv/einsum/Einsum«
'gat_conv/einsum_1/Einsum/ReadVariableOpReadVariableOp0gat_conv_einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02)
'gat_conv/einsum_1/Einsum/ReadVariableOpц
gat_conv/einsum_1/EinsumEinsumgat_conv/einsum/Einsum:output:0/gat_conv/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv/einsum_1/Einsum«
'gat_conv/einsum_2/Einsum/ReadVariableOpReadVariableOp0gat_conv_einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02)
'gat_conv/einsum_2/Einsum/ReadVariableOpц
gat_conv/einsum_2/EinsumEinsumgat_conv/einsum/Einsum:output:0/gat_conv/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv/einsum_2/Einsum√
gat_conv/einsum_3/EinsumEinsum!gat_conv/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
gat_conv/einsum_3/EinsumЂ
gat_conv/add_1AddV2!gat_conv/einsum_1/Einsum:output:0!gat_conv/einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/add_1|
gat_conv/LeakyRelu	LeakyRelugat_conv/add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/LeakyRelu]

gat_conv/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

gat_conv/x≤
gat_conv/EqualEqualgat_conv/set_diag:output:0gat_conv/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
gat_conv/Equalo
gat_conv/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2
gat_conv/SelectV2/to
gat_conv/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gat_conv/SelectV2/eЇ
gat_conv/SelectV2SelectV2gat_conv/Equal:z:0gat_conv/SelectV2/t:output:0gat_conv/SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv/SelectV2Х
gat_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
gat_conv/strided_slice_1/stackЩ
 gat_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 gat_conv/strided_slice_1/stack_1Щ
 gat_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 gat_conv/strided_slice_1/stack_2ф
gat_conv/strided_slice_1StridedSlicegat_conv/SelectV2:output:0'gat_conv/strided_slice_1/stack:output:0)gat_conv/strided_slice_1/stack_1:output:0)gat_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
gat_conv/strided_slice_1™
gat_conv/add_2AddV2 gat_conv/LeakyRelu:activations:0!gat_conv/strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/add_2
gat_conv/SoftmaxSoftmaxgat_conv/add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/SoftmaxЕ
gat_conv/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gat_conv/dropout/dropout/Constƒ
gat_conv/dropout/dropout/MulMulgat_conv/Softmax:softmax:0'gat_conv/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/dropout/dropout/MulК
gat_conv/dropout/dropout/ShapeShapegat_conv/Softmax:softmax:0*
T0*
_output_shapes
:2 
gat_conv/dropout/dropout/Shapeс
5gat_conv/dropout/dropout/random_uniform/RandomUniformRandomUniform'gat_conv/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype027
5gat_conv/dropout/dropout/random_uniform/RandomUniformЧ
'gat_conv/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'gat_conv/dropout/dropout/GreaterEqual/yМ
%gat_conv/dropout/dropout/GreaterEqualGreaterEqual>gat_conv/dropout/dropout/random_uniform/RandomUniform:output:00gat_conv/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2'
%gat_conv/dropout/dropout/GreaterEqualЉ
gat_conv/dropout/dropout/CastCast)gat_conv/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/dropout/dropout/Cast»
gat_conv/dropout/dropout/Mul_1Mul gat_conv/dropout/dropout/Mul:z:0!gat_conv/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2 
gat_conv/dropout/dropout/Mul_1м
gat_conv/einsum_4/EinsumEinsum"gat_conv/dropout/dropout/Mul_1:z:0gat_conv/einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
gat_conv/einsum_4/Einsumu
gat_conv/Shape_1Shape!gat_conv/einsum_4/Einsum:output:0*
T0*
_output_shapes
:2
gat_conv/Shape_1К
gat_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gat_conv/strided_slice_2/stackЧ
 gat_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2"
 gat_conv/strided_slice_2/stack_1О
 gat_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gat_conv/strided_slice_2/stack_2Ґ
gat_conv/strided_slice_2StridedSlicegat_conv/Shape_1:output:0'gat_conv/strided_slice_2/stack:output:0)gat_conv/strided_slice_2/stack_1:output:0)gat_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv/strided_slice_2~
gat_conv/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
gat_conv/concat/values_1n
gat_conv/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv/concat/axisЅ
gat_conv/concatConcatV2!gat_conv/strided_slice_2:output:0!gat_conv/concat/values_1:output:0gat_conv/concat/axis:output:0*
N*
T0*
_output_shapes
:2
gat_conv/concat£
gat_conv/ReshapeReshape!gat_conv/einsum_4/Einsum:output:0gat_conv/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv/Reshape°
gat_conv/add_3/ReadVariableOpReadVariableOp&gat_conv_add_3_readvariableop_resource*
_output_shapes
:d*
dtype02
gat_conv/add_3/ReadVariableOpҐ
gat_conv/add_3AddV2gat_conv/Reshape:output:0%gat_conv/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv/add_3\
gat_conv_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2
gat_conv_1/ShapeК
gat_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gat_conv_1/strided_slice/stackЧ
 gat_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2"
 gat_conv_1/strided_slice/stack_1О
 gat_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gat_conv_1/strided_slice/stack_2Ґ
gat_conv_1/strided_sliceStridedSlicegat_conv_1/Shape:output:0'gat_conv_1/strided_slice/stack:output:0)gat_conv_1/strided_slice/stack_1:output:0)gat_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv_1/strided_sliceП
 gat_conv_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 gat_conv_1/Sum/reduction_indicesП
gat_conv_1/SumSuminputs_1)gat_conv_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv_1/SumУ
"gat_conv_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2$
"gat_conv_1/Sum_1/reduction_indicesХ
gat_conv_1/Sum_1Suminputs_1+gat_conv_1/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv_1/Sum_1Р
gat_conv_1/addAddV2gat_conv_1/Sum_1:output:0gat_conv_1/Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv_1/addp
gat_conv_1/set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv_1/set_diag/k≥
gat_conv_1/set_diagMatrixSetDiagV3inputs_1gat_conv_1/add:z:0gat_conv_1/set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv_1/set_diag«
'gat_conv_1/einsum/Einsum/ReadVariableOpReadVariableOp0gat_conv_1_einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02)
'gat_conv_1/einsum/Einsum/ReadVariableOpи
gat_conv_1/einsum/EinsumEinsumgat_conv/add_3:z:0/gat_conv_1/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
gat_conv_1/einsum/EinsumЌ
)gat_conv_1/einsum_1/Einsum/ReadVariableOpReadVariableOp2gat_conv_1_einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02+
)gat_conv_1/einsum_1/Einsum/ReadVariableOpю
gat_conv_1/einsum_1/EinsumEinsum!gat_conv_1/einsum/Einsum:output:01gat_conv_1/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv_1/einsum_1/EinsumЌ
)gat_conv_1/einsum_2/Einsum/ReadVariableOpReadVariableOp2gat_conv_1_einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02+
)gat_conv_1/einsum_2/Einsum/ReadVariableOpю
gat_conv_1/einsum_2/EinsumEinsum!gat_conv_1/einsum/Einsum:output:01gat_conv_1/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv_1/einsum_2/Einsum…
gat_conv_1/einsum_3/EinsumEinsum#gat_conv_1/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
gat_conv_1/einsum_3/Einsum≥
gat_conv_1/add_1AddV2#gat_conv_1/einsum_1/Einsum:output:0#gat_conv_1/einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/add_1В
gat_conv_1/LeakyRelu	LeakyRelugat_conv_1/add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/LeakyRelua
gat_conv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gat_conv_1/xЇ
gat_conv_1/EqualEqualgat_conv_1/set_diag:output:0gat_conv_1/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
gat_conv_1/Equals
gat_conv_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2
gat_conv_1/SelectV2/ts
gat_conv_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gat_conv_1/SelectV2/eƒ
gat_conv_1/SelectV2SelectV2gat_conv_1/Equal:z:0gat_conv_1/SelectV2/t:output:0gat_conv_1/SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv_1/SelectV2Щ
 gat_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2"
 gat_conv_1/strided_slice_1/stackЭ
"gat_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2$
"gat_conv_1/strided_slice_1/stack_1Э
"gat_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2$
"gat_conv_1/strided_slice_1/stack_2А
gat_conv_1/strided_slice_1StridedSlicegat_conv_1/SelectV2:output:0)gat_conv_1/strided_slice_1/stack:output:0+gat_conv_1/strided_slice_1/stack_1:output:0+gat_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
gat_conv_1/strided_slice_1≤
gat_conv_1/add_2AddV2"gat_conv_1/LeakyRelu:activations:0#gat_conv_1/strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/add_2Е
gat_conv_1/SoftmaxSoftmaxgat_conv_1/add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/SoftmaxЙ
 gat_conv_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gat_conv_1/dropout/dropout/Constћ
gat_conv_1/dropout/dropout/MulMulgat_conv_1/Softmax:softmax:0)gat_conv_1/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2 
gat_conv_1/dropout/dropout/MulР
 gat_conv_1/dropout/dropout/ShapeShapegat_conv_1/Softmax:softmax:0*
T0*
_output_shapes
:2"
 gat_conv_1/dropout/dropout/Shapeч
7gat_conv_1/dropout/dropout/random_uniform/RandomUniformRandomUniform)gat_conv_1/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype029
7gat_conv_1/dropout/dropout/random_uniform/RandomUniformЫ
)gat_conv_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gat_conv_1/dropout/dropout/GreaterEqual/yФ
'gat_conv_1/dropout/dropout/GreaterEqualGreaterEqual@gat_conv_1/dropout/dropout/random_uniform/RandomUniform:output:02gat_conv_1/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2)
'gat_conv_1/dropout/dropout/GreaterEqual¬
gat_conv_1/dropout/dropout/CastCast+gat_conv_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2!
gat_conv_1/dropout/dropout/Cast–
 gat_conv_1/dropout/dropout/Mul_1Mul"gat_conv_1/dropout/dropout/Mul:z:0#gat_conv_1/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2"
 gat_conv_1/dropout/dropout/Mul_1ф
gat_conv_1/einsum_4/EinsumEinsum$gat_conv_1/dropout/dropout/Mul_1:z:0!gat_conv_1/einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
gat_conv_1/einsum_4/Einsum{
gat_conv_1/Shape_1Shape#gat_conv_1/einsum_4/Einsum:output:0*
T0*
_output_shapes
:2
gat_conv_1/Shape_1О
 gat_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gat_conv_1/strided_slice_2/stackЫ
"gat_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2$
"gat_conv_1/strided_slice_2/stack_1Т
"gat_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gat_conv_1/strided_slice_2/stack_2Ѓ
gat_conv_1/strided_slice_2StridedSlicegat_conv_1/Shape_1:output:0)gat_conv_1/strided_slice_2/stack:output:0+gat_conv_1/strided_slice_2/stack_1:output:0+gat_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv_1/strided_slice_2В
gat_conv_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
gat_conv_1/concat/values_1r
gat_conv_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv_1/concat/axisЋ
gat_conv_1/concatConcatV2#gat_conv_1/strided_slice_2:output:0#gat_conv_1/concat/values_1:output:0gat_conv_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
gat_conv_1/concatЂ
gat_conv_1/ReshapeReshape#gat_conv_1/einsum_4/Einsum:output:0gat_conv_1/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv_1/ReshapeІ
gat_conv_1/add_3/ReadVariableOpReadVariableOp(gat_conv_1_add_3_readvariableop_resource*
_output_shapes
:d*
dtype02!
gat_conv_1/add_3/ReadVariableOp™
gat_conv_1/add_3AddV2gat_conv_1/Reshape:output:0'gat_conv_1/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv_1/add_3®
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freer
dense/Tensordot/ShapeShapegat_conv_1/add_3:z:0*
T0*
_output_shapes
:2
dense/Tensordot/ShapeА
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisп
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2Д
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisх
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/ConstШ
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1†
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisќ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat§
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack±
dense/Tensordot/transpose	Transposegat_conv_1/add_3:z:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
dense/Tensordot/transposeЈ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense/Tensordot/Reshapeґ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2А
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisџ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1©
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
dense/TensordotЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp†
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
dense/BiasAddx
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
dense/Softmaxw
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityќ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^gat_conv/add_3/ReadVariableOp&^gat_conv/einsum/Einsum/ReadVariableOp(^gat_conv/einsum_1/Einsum/ReadVariableOp(^gat_conv/einsum_2/Einsum/ReadVariableOp ^gat_conv_1/add_3/ReadVariableOp(^gat_conv_1/einsum/Einsum/ReadVariableOp*^gat_conv_1/einsum_1/Einsum/ReadVariableOp*^gat_conv_1/einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2>
gat_conv/add_3/ReadVariableOpgat_conv/add_3/ReadVariableOp2N
%gat_conv/einsum/Einsum/ReadVariableOp%gat_conv/einsum/Einsum/ReadVariableOp2R
'gat_conv/einsum_1/Einsum/ReadVariableOp'gat_conv/einsum_1/Einsum/ReadVariableOp2R
'gat_conv/einsum_2/Einsum/ReadVariableOp'gat_conv/einsum_2/Einsum/ReadVariableOp2B
gat_conv_1/add_3/ReadVariableOpgat_conv_1/add_3/ReadVariableOp2R
'gat_conv_1/einsum/Einsum/ReadVariableOp'gat_conv_1/einsum/Einsum/ReadVariableOp2V
)gat_conv_1/einsum_1/Einsum/ReadVariableOp)gat_conv_1/einsum_1/Einsum/ReadVariableOp2V
)gat_conv_1/einsum_2/Einsum/ReadVariableOp)gat_conv_1/einsum_2/Einsum/ReadVariableOp:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1
ІE
—
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_3590

inputs
inputs_1;
%einsum_einsum_readvariableop_resource:d

=
'einsum_1_einsum_readvariableop_resource:

=
'einsum_2_einsum_readvariableop_resource:

+
add_3_readvariableop_resource:d
identityИҐadd_3/ReadVariableOpҐeinsum/Einsum/ReadVariableOpҐeinsum_1/Einsum/ReadVariableOpҐeinsum_2/Einsum/ReadVariableOpF
ShapeShapeinputs_1*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackБ
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2а
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicey
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2
Sum/reduction_indicesn
SumSuminputs_1Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum}
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
Sum_1/reduction_indicest
Sum_1Suminputs_1 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
Sum_1d
addAddV2Sum_1:output:0Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
addZ

set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2

set_diag/kЗ
set_diagMatrixSetDiagV3inputs_1add:z:0set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

set_diag¶
einsum/Einsum/ReadVariableOpReadVariableOp%einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02
einsum/Einsum/ReadVariableOpї
einsum/EinsumEinsuminputs$einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
einsum/Einsumђ
einsum_1/Einsum/ReadVariableOpReadVariableOp'einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_1/Einsum/ReadVariableOp“
einsum_1/EinsumEinsumeinsum/Einsum:output:0&einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_1/Einsumђ
einsum_2/Einsum/ReadVariableOpReadVariableOp'einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02 
einsum_2/Einsum/ReadVariableOp“
einsum_2/EinsumEinsumeinsum/Einsum:output:0&einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
einsum_2/Einsum®
einsum_3/EinsumEinsumeinsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
einsum_3/EinsumЗ
add_1AddV2einsum_1/Einsum:output:0einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_1a
	LeakyRelu	LeakyRelu	add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
	LeakyReluK
xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
xО
EqualEqualset_diag:output:0
x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
Equal]

SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2

SelectV2/t]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2

SelectV2/eН
SelectV2SelectV2	Equal:z:0SelectV2/t:output:0SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2

SelectV2Г
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stackЗ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice_1/stack_1З
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice_1/stack_2Њ
strided_slice_1StridedSliceSelectV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
strided_slice_1Ж
add_2AddV2LeakyRelu:activations:0strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
add_2d
SoftmaxSoftmax	add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2	
Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const†
dropout/dropout/MulMulSoftmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mulo
dropout/dropout/ShapeShapeSoftmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
dropout/dropout/Mul_1»
einsum_4/EinsumEinsumdropout/dropout/Mul_1:z:0einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
einsum_4/EinsumZ
Shape_1Shapeeinsum_4/Einsum:output:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stackЕ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape_1:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_2l
concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
concat/values_1\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axisФ
concatConcatV2strided_slice_2:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat
ReshapeReshapeeinsum_4/Einsum:output:0concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2	
ReshapeЖ
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:d*
dtype02
add_3/ReadVariableOp~
add_3AddV2Reshape:output:0add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
add_3i
IdentityIdentity	add_3:z:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Фd2

Identity∆
NoOpNoOp^add_3/ReadVariableOp^einsum/Einsum/ReadVariableOp^einsum_1/Einsum/ReadVariableOp^einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€Фd:€€€€€€€€€ФФ: : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2<
einsum/Einsum/ReadVariableOpeinsum/Einsum/ReadVariableOp2@
einsum_1/Einsum/ReadVariableOpeinsum_1/Einsum/ReadVariableOp2@
einsum_2/Einsum/ReadVariableOpeinsum_2/Einsum/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Фd
 
_user_specified_nameinputs:UQ
-
_output_shapes
:€€€€€€€€€ФФ
 
_user_specified_nameinputs
тƒ
€
?__inference_model_layer_call_and_return_conditional_losses_4175
inputs_0
inputs_1E
.gat_conv_einsum_einsum_readvariableop_resource:Щ

F
0gat_conv_einsum_1_einsum_readvariableop_resource:

F
0gat_conv_einsum_2_einsum_readvariableop_resource:

4
&gat_conv_add_3_readvariableop_resource:dF
0gat_conv_1_einsum_einsum_readvariableop_resource:d

H
2gat_conv_1_einsum_1_einsum_readvariableop_resource:

H
2gat_conv_1_einsum_2_einsum_readvariableop_resource:

6
(gat_conv_1_add_3_readvariableop_resource:d9
'dense_tensordot_readvariableop_resource:d3
%dense_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐgat_conv/add_3/ReadVariableOpҐ%gat_conv/einsum/Einsum/ReadVariableOpҐ'gat_conv/einsum_1/Einsum/ReadVariableOpҐ'gat_conv/einsum_2/Einsum/ReadVariableOpҐgat_conv_1/add_3/ReadVariableOpҐ'gat_conv_1/einsum/Einsum/ReadVariableOpҐ)gat_conv_1/einsum_1/Einsum/ReadVariableOpҐ)gat_conv_1/einsum_2/Einsum/ReadVariableOpX
gat_conv/ShapeShapeinputs_1*
T0*
_output_shapes
:2
gat_conv/ShapeЖ
gat_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gat_conv/strided_slice/stackУ
gat_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2 
gat_conv/strided_slice/stack_1К
gat_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gat_conv/strided_slice/stack_2Ц
gat_conv/strided_sliceStridedSlicegat_conv/Shape:output:0%gat_conv/strided_slice/stack:output:0'gat_conv/strided_slice/stack_1:output:0'gat_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv/strided_sliceЛ
gat_conv/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2 
gat_conv/Sum/reduction_indicesЙ
gat_conv/SumSuminputs_1'gat_conv/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv/SumП
 gat_conv/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2"
 gat_conv/Sum_1/reduction_indicesП
gat_conv/Sum_1Suminputs_1)gat_conv/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv/Sum_1И
gat_conv/addAddV2gat_conv/Sum_1:output:0gat_conv/Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv/addl
gat_conv/set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv/set_diag/kЂ
gat_conv/set_diagMatrixSetDiagV3inputs_1gat_conv/add:z:0gat_conv/set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv/set_diag¬
%gat_conv/einsum/Einsum/ReadVariableOpReadVariableOp.gat_conv_einsum_einsum_readvariableop_resource*#
_output_shapes
:Щ

*
dtype02'
%gat_conv/einsum/Einsum/ReadVariableOpЎ
gat_conv/einsum/EinsumEinsuminputs_0-gat_conv/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
gat_conv/einsum/Einsum«
'gat_conv/einsum_1/Einsum/ReadVariableOpReadVariableOp0gat_conv_einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02)
'gat_conv/einsum_1/Einsum/ReadVariableOpц
gat_conv/einsum_1/EinsumEinsumgat_conv/einsum/Einsum:output:0/gat_conv/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv/einsum_1/Einsum«
'gat_conv/einsum_2/Einsum/ReadVariableOpReadVariableOp0gat_conv_einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02)
'gat_conv/einsum_2/Einsum/ReadVariableOpц
gat_conv/einsum_2/EinsumEinsumgat_conv/einsum/Einsum:output:0/gat_conv/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv/einsum_2/Einsum√
gat_conv/einsum_3/EinsumEinsum!gat_conv/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
gat_conv/einsum_3/EinsumЂ
gat_conv/add_1AddV2!gat_conv/einsum_1/Einsum:output:0!gat_conv/einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/add_1|
gat_conv/LeakyRelu	LeakyRelugat_conv/add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/LeakyRelu]

gat_conv/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

gat_conv/x≤
gat_conv/EqualEqualgat_conv/set_diag:output:0gat_conv/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
gat_conv/Equalo
gat_conv/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2
gat_conv/SelectV2/to
gat_conv/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gat_conv/SelectV2/eЇ
gat_conv/SelectV2SelectV2gat_conv/Equal:z:0gat_conv/SelectV2/t:output:0gat_conv/SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv/SelectV2Х
gat_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2 
gat_conv/strided_slice_1/stackЩ
 gat_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2"
 gat_conv/strided_slice_1/stack_1Щ
 gat_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2"
 gat_conv/strided_slice_1/stack_2ф
gat_conv/strided_slice_1StridedSlicegat_conv/SelectV2:output:0'gat_conv/strided_slice_1/stack:output:0)gat_conv/strided_slice_1/stack_1:output:0)gat_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
gat_conv/strided_slice_1™
gat_conv/add_2AddV2 gat_conv/LeakyRelu:activations:0!gat_conv/strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/add_2
gat_conv/SoftmaxSoftmaxgat_conv/add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/SoftmaxЕ
gat_conv/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2 
gat_conv/dropout/dropout/Constƒ
gat_conv/dropout/dropout/MulMulgat_conv/Softmax:softmax:0'gat_conv/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/dropout/dropout/MulК
gat_conv/dropout/dropout/ShapeShapegat_conv/Softmax:softmax:0*
T0*
_output_shapes
:2 
gat_conv/dropout/dropout/Shapeс
5gat_conv/dropout/dropout/random_uniform/RandomUniformRandomUniform'gat_conv/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype027
5gat_conv/dropout/dropout/random_uniform/RandomUniformЧ
'gat_conv/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'gat_conv/dropout/dropout/GreaterEqual/yМ
%gat_conv/dropout/dropout/GreaterEqualGreaterEqual>gat_conv/dropout/dropout/random_uniform/RandomUniform:output:00gat_conv/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2'
%gat_conv/dropout/dropout/GreaterEqualЉ
gat_conv/dropout/dropout/CastCast)gat_conv/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv/dropout/dropout/Cast»
gat_conv/dropout/dropout/Mul_1Mul gat_conv/dropout/dropout/Mul:z:0!gat_conv/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2 
gat_conv/dropout/dropout/Mul_1м
gat_conv/einsum_4/EinsumEinsum"gat_conv/dropout/dropout/Mul_1:z:0gat_conv/einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
gat_conv/einsum_4/Einsumu
gat_conv/Shape_1Shape!gat_conv/einsum_4/Einsum:output:0*
T0*
_output_shapes
:2
gat_conv/Shape_1К
gat_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gat_conv/strided_slice_2/stackЧ
 gat_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2"
 gat_conv/strided_slice_2/stack_1О
 gat_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gat_conv/strided_slice_2/stack_2Ґ
gat_conv/strided_slice_2StridedSlicegat_conv/Shape_1:output:0'gat_conv/strided_slice_2/stack:output:0)gat_conv/strided_slice_2/stack_1:output:0)gat_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv/strided_slice_2~
gat_conv/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
gat_conv/concat/values_1n
gat_conv/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv/concat/axisЅ
gat_conv/concatConcatV2!gat_conv/strided_slice_2:output:0!gat_conv/concat/values_1:output:0gat_conv/concat/axis:output:0*
N*
T0*
_output_shapes
:2
gat_conv/concat£
gat_conv/ReshapeReshape!gat_conv/einsum_4/Einsum:output:0gat_conv/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv/Reshape°
gat_conv/add_3/ReadVariableOpReadVariableOp&gat_conv_add_3_readvariableop_resource*
_output_shapes
:d*
dtype02
gat_conv/add_3/ReadVariableOpҐ
gat_conv/add_3AddV2gat_conv/Reshape:output:0%gat_conv/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv/add_3\
gat_conv_1/ShapeShapeinputs_1*
T0*
_output_shapes
:2
gat_conv_1/ShapeК
gat_conv_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gat_conv_1/strided_slice/stackЧ
 gat_conv_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2"
 gat_conv_1/strided_slice/stack_1О
 gat_conv_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gat_conv_1/strided_slice/stack_2Ґ
gat_conv_1/strided_sliceStridedSlicegat_conv_1/Shape:output:0'gat_conv_1/strided_slice/stack:output:0)gat_conv_1/strided_slice/stack_1:output:0)gat_conv_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv_1/strided_sliceП
 gat_conv_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2"
 gat_conv_1/Sum/reduction_indicesП
gat_conv_1/SumSuminputs_1)gat_conv_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv_1/SumУ
"gat_conv_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2$
"gat_conv_1/Sum_1/reduction_indicesХ
gat_conv_1/Sum_1Suminputs_1+gat_conv_1/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv_1/Sum_1Р
gat_conv_1/addAddV2gat_conv_1/Sum_1:output:0gat_conv_1/Sum:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ф2
gat_conv_1/addp
gat_conv_1/set_diag/kConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv_1/set_diag/k≥
gat_conv_1/set_diagMatrixSetDiagV3inputs_1gat_conv_1/add:z:0gat_conv_1/set_diag/k:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv_1/set_diag«
'gat_conv_1/einsum/Einsum/ReadVariableOpReadVariableOp0gat_conv_1_einsum_einsum_readvariableop_resource*"
_output_shapes
:d

*
dtype02)
'gat_conv_1/einsum/Einsum/ReadVariableOpи
gat_conv_1/einsum/EinsumEinsumgat_conv/add_3:z:0/gat_conv_1/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*
equation...NI,IHO->...NHO2
gat_conv_1/einsum/EinsumЌ
)gat_conv_1/einsum_1/Einsum/ReadVariableOpReadVariableOp2gat_conv_1_einsum_1_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02+
)gat_conv_1/einsum_1/Einsum/ReadVariableOpю
gat_conv_1/einsum_1/EinsumEinsum!gat_conv_1/einsum/Einsum:output:01gat_conv_1/einsum_1/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv_1/einsum_1/EinsumЌ
)gat_conv_1/einsum_2/Einsum/ReadVariableOpReadVariableOp2gat_conv_1_einsum_2_einsum_readvariableop_resource*"
_output_shapes
:

*
dtype02+
)gat_conv_1/einsum_2/Einsum/ReadVariableOpю
gat_conv_1/einsum_2/EinsumEinsum!gat_conv_1/einsum/Einsum:output:01gat_conv_1/einsum_2/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф
* 
equation...NHI,IHO->...NHO2
gat_conv_1/einsum_2/Einsum…
gat_conv_1/einsum_3/EinsumEinsum#gat_conv_1/einsum_2/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€
Ф*
equation...ABC->...CBA2
gat_conv_1/einsum_3/Einsum≥
gat_conv_1/add_1AddV2#gat_conv_1/einsum_1/Einsum:output:0#gat_conv_1/einsum_3/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/add_1В
gat_conv_1/LeakyRelu	LeakyRelugat_conv_1/add_1:z:0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/LeakyRelua
gat_conv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gat_conv_1/xЇ
gat_conv_1/EqualEqualgat_conv_1/set_diag:output:0gat_conv_1/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ*
incompatible_shape_error( 2
gat_conv_1/Equals
gat_conv_1/SelectV2/tConst*
_output_shapes
: *
dtype0*
valueB
 *щ–2
gat_conv_1/SelectV2/ts
gat_conv_1/SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gat_conv_1/SelectV2/eƒ
gat_conv_1/SelectV2SelectV2gat_conv_1/Equal:z:0gat_conv_1/SelectV2/t:output:0gat_conv_1/SelectV2/e:output:0*
T0*-
_output_shapes
:€€€€€€€€€ФФ2
gat_conv_1/SelectV2Щ
 gat_conv_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2"
 gat_conv_1/strided_slice_1/stackЭ
"gat_conv_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2$
"gat_conv_1/strided_slice_1/stack_1Э
"gat_conv_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2$
"gat_conv_1/strided_slice_1/stack_2А
gat_conv_1/strided_slice_1StridedSlicegat_conv_1/SelectV2:output:0)gat_conv_1/strided_slice_1/stack:output:0+gat_conv_1/strided_slice_1/stack_1:output:0+gat_conv_1/strided_slice_1/stack_2:output:0*
Index0*
T0*1
_output_shapes
:€€€€€€€€€ФФ*

begin_mask*
ellipsis_mask*
end_mask*
new_axis_mask2
gat_conv_1/strided_slice_1≤
gat_conv_1/add_2AddV2"gat_conv_1/LeakyRelu:activations:0#gat_conv_1/strided_slice_1:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/add_2Е
gat_conv_1/SoftmaxSoftmaxgat_conv_1/add_2:z:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2
gat_conv_1/SoftmaxЙ
 gat_conv_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2"
 gat_conv_1/dropout/dropout/Constћ
gat_conv_1/dropout/dropout/MulMulgat_conv_1/Softmax:softmax:0)gat_conv_1/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2 
gat_conv_1/dropout/dropout/MulР
 gat_conv_1/dropout/dropout/ShapeShapegat_conv_1/Softmax:softmax:0*
T0*
_output_shapes
:2"
 gat_conv_1/dropout/dropout/Shapeч
7gat_conv_1/dropout/dropout/random_uniform/RandomUniformRandomUniform)gat_conv_1/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф*
dtype029
7gat_conv_1/dropout/dropout/random_uniform/RandomUniformЫ
)gat_conv_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2+
)gat_conv_1/dropout/dropout/GreaterEqual/yФ
'gat_conv_1/dropout/dropout/GreaterEqualGreaterEqual@gat_conv_1/dropout/dropout/random_uniform/RandomUniform:output:02gat_conv_1/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2)
'gat_conv_1/dropout/dropout/GreaterEqual¬
gat_conv_1/dropout/dropout/CastCast+gat_conv_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€Ф
Ф2!
gat_conv_1/dropout/dropout/Cast–
 gat_conv_1/dropout/dropout/Mul_1Mul"gat_conv_1/dropout/dropout/Mul:z:0#gat_conv_1/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€Ф
Ф2"
 gat_conv_1/dropout/dropout/Mul_1ф
gat_conv_1/einsum_4/EinsumEinsum$gat_conv_1/dropout/dropout/Mul_1:z:0!gat_conv_1/einsum/Einsum:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Ф

*#
equation...NHM,...MHI->...NHI2
gat_conv_1/einsum_4/Einsum{
gat_conv_1/Shape_1Shape#gat_conv_1/einsum_4/Einsum:output:0*
T0*
_output_shapes
:2
gat_conv_1/Shape_1О
 gat_conv_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 gat_conv_1/strided_slice_2/stackЫ
"gat_conv_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ю€€€€€€€€2$
"gat_conv_1/strided_slice_2/stack_1Т
"gat_conv_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"gat_conv_1/strided_slice_2/stack_2Ѓ
gat_conv_1/strided_slice_2StridedSlicegat_conv_1/Shape_1:output:0)gat_conv_1/strided_slice_2/stack:output:0+gat_conv_1/strided_slice_2/stack_1:output:0+gat_conv_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gat_conv_1/strided_slice_2В
gat_conv_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:d2
gat_conv_1/concat/values_1r
gat_conv_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
gat_conv_1/concat/axisЋ
gat_conv_1/concatConcatV2#gat_conv_1/strided_slice_2:output:0#gat_conv_1/concat/values_1:output:0gat_conv_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
gat_conv_1/concatЂ
gat_conv_1/ReshapeReshape#gat_conv_1/einsum_4/Einsum:output:0gat_conv_1/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv_1/ReshapeІ
gat_conv_1/add_3/ReadVariableOpReadVariableOp(gat_conv_1_add_3_readvariableop_resource*
_output_shapes
:d*
dtype02!
gat_conv_1/add_3/ReadVariableOp™
gat_conv_1/add_3AddV2gat_conv_1/Reshape:output:0'gat_conv_1/add_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
gat_conv_1/add_3®
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freer
dense/Tensordot/ShapeShapegat_conv_1/add_3:z:0*
T0*
_output_shapes
:2
dense/Tensordot/ShapeА
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisп
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2Д
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisх
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/ConstШ
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1†
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisќ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat§
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack±
dense/Tensordot/transpose	Transposegat_conv_1/add_3:z:0dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Фd2
dense/Tensordot/transposeЈ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense/Tensordot/Reshapeґ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2А
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisџ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1©
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
dense/TensordotЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp†
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
dense/BiasAddx
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Ф2
dense/Softmaxw
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*,
_output_shapes
:€€€€€€€€€Ф2

Identityќ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^gat_conv/add_3/ReadVariableOp&^gat_conv/einsum/Einsum/ReadVariableOp(^gat_conv/einsum_1/Einsum/ReadVariableOp(^gat_conv/einsum_2/Einsum/ReadVariableOp ^gat_conv_1/add_3/ReadVariableOp(^gat_conv_1/einsum/Einsum/ReadVariableOp*^gat_conv_1/einsum_1/Einsum/ReadVariableOp*^gat_conv_1/einsum_2/Einsum/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:€€€€€€€€€ФЩ:€€€€€€€€€ФФ: : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2>
gat_conv/add_3/ReadVariableOpgat_conv/add_3/ReadVariableOp2N
%gat_conv/einsum/Einsum/ReadVariableOp%gat_conv/einsum/Einsum/ReadVariableOp2R
'gat_conv/einsum_1/Einsum/ReadVariableOp'gat_conv/einsum_1/Einsum/ReadVariableOp2R
'gat_conv/einsum_2/Einsum/ReadVariableOp'gat_conv/einsum_2/Einsum/ReadVariableOp2B
gat_conv_1/add_3/ReadVariableOpgat_conv_1/add_3/ReadVariableOp2R
'gat_conv_1/einsum/Einsum/ReadVariableOp'gat_conv_1/einsum/Einsum/ReadVariableOp2V
)gat_conv_1/einsum_1/Einsum/ReadVariableOp)gat_conv_1/einsum_1/Einsum/ReadVariableOp2V
)gat_conv_1/einsum_2/Einsum/ReadVariableOp)gat_conv_1/einsum_2/Einsum/ReadVariableOp:W S
-
_output_shapes
:€€€€€€€€€ФЩ
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:€€€€€€€€€ФФ
"
_user_specified_name
inputs/1"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ц
serving_defaultв
A
input_16
serving_default_input_1:0€€€€€€€€€ФЩ
A
input_26
serving_default_input_2:0€€€€€€€€€ФФ>
dense5
StatefulPartitionedCall:0€€€€€€€€€Фtensorflow/serving/predict:Ёx
т
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
*l&call_and_return_all_conditional_losses
m_default_save_signature
n__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ю
kwargs_keys

kernel
attn_kernel_self
attn_kernel_neigh
attn_kernel_neighs
bias
dropout
trainable_variables
regularization_losses
	variables
	keras_api
*o&call_and_return_all_conditional_losses
p__call__"
_tf_keras_layer
Ю
kwargs_keys

kernel
attn_kernel_self
attn_kernel_neigh
attn_kernel_neighs
bias
dropout
trainable_variables
regularization_losses
	variables
	keras_api
*q&call_and_return_all_conditional_losses
r__call__"
_tf_keras_layer
ї

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
*s&call_and_return_all_conditional_losses
t__call__"
_tf_keras_layer
ј
&iter
	'decay
(learning_rate
)momentum
*rho	rmsb	rmsc	rmsd	rmse	rmsf	rmsg	rmsh	rmsi	 rmsj	!rmsk"
	optimizer
f
0
1
2
3
4
5
6
7
 8
!9"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
 8
!9"
trackable_list_wrapper
 
trainable_variables

+layers
regularization_losses
,metrics
		variables
-non_trainable_variables
.layer_regularization_losses
/layer_metrics
n__call__
m_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
userving_default"
signature_map
 "
trackable_list_wrapper
&:$Щ

2gat_conv/kernel
/:-

2gat_conv/attn_kernel_self
0:.

2gat_conv/attn_kernel_neigh
:d2gat_conv/bias
•
0trainable_variables
1regularization_losses
2	variables
3	keras_api
*v&call_and_return_all_conditional_losses
w__call__"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
≠
trainable_variables

4layers
regularization_losses
5metrics
	variables
6non_trainable_variables
7layer_regularization_losses
8layer_metrics
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%d

2gat_conv_1/kernel
1:/

2gat_conv_1/attn_kernel_self
2:0

2gat_conv_1/attn_kernel_neigh
:d2gat_conv_1/bias
•
9trainable_variables
:regularization_losses
;	variables
<	keras_api
*x&call_and_return_all_conditional_losses
y__call__"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
≠
trainable_variables

=layers
regularization_losses
>metrics
	variables
?non_trainable_variables
@layer_regularization_losses
Alayer_metrics
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:d2dense/kernel
:2
dense/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
≠
"trainable_variables

Blayers
#regularization_losses
Cmetrics
$	variables
Dnon_trainable_variables
Elayer_regularization_losses
Flayer_metrics
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
C
0
1
2
3
4"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
0trainable_variables

Jlayers
1regularization_losses
Kmetrics
2	variables
Lnon_trainable_variables
Mlayer_regularization_losses
Nlayer_metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
9trainable_variables

Olayers
:regularization_losses
Pmetrics
;	variables
Qnon_trainable_variables
Rlayer_regularization_losses
Slayer_metrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ttotal
	Ucount
V	variables
W	keras_api"
_tf_keras_metric
^
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api"
_tf_keras_metric
q
]
thresholds
^true_positives
_false_positives
`	variables
a	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
T0
U1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
^0
_1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
0:.Щ

2RMSprop/gat_conv/kernel/rms
9:7

2%RMSprop/gat_conv/attn_kernel_self/rms
::8

2&RMSprop/gat_conv/attn_kernel_neigh/rms
%:#d2RMSprop/gat_conv/bias/rms
1:/d

2RMSprop/gat_conv_1/kernel/rms
;:9

2'RMSprop/gat_conv_1/attn_kernel_self/rms
<::

2(RMSprop/gat_conv_1/attn_kernel_neigh/rms
':%d2RMSprop/gat_conv_1/bias/rms
(:&d2RMSprop/dense/kernel/rms
": 2RMSprop/dense/bias/rms
 2«
?__inference_model_layer_call_and_return_conditional_losses_4031
?__inference_model_layer_call_and_return_conditional_losses_4175
?__inference_model_layer_call_and_return_conditional_losses_3825
?__inference_model_layer_call_and_return_conditional_losses_3853ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”B–
__inference__wrapped_model_3289input_1input_2"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ё2џ
$__inference_model_layer_call_fn_3501
$__inference_model_layer_call_fn_4201
$__inference_model_layer_call_fn_4227
$__inference_model_layer_call_fn_3797ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 2«
B__inference_gat_conv_layer_call_and_return_conditional_losses_4288
B__inference_gat_conv_layer_call_and_return_conditional_losses_4349Љ
µ≤±
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
Ф2С
'__inference_gat_conv_layer_call_fn_4363
'__inference_gat_conv_layer_call_fn_4377Љ
µ≤±
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ќ2Ћ
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_4438
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_4499Љ
µ≤±
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
Ш2Х
)__inference_gat_conv_1_layer_call_fn_4513
)__inference_gat_conv_1_layer_call_fn_4527Љ
µ≤±
FullArgSpec
argsЪ

jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
й2ж
?__inference_dense_layer_call_and_return_conditional_losses_4558Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
$__inference_dense_layer_call_fn_4567Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–BЌ
"__inference_signature_wrapper_3887input_1input_2"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
  
__inference__wrapped_model_3289¶
 !dҐa
ZҐW
UЪR
'К$
input_1€€€€€€€€€ФЩ
'К$
input_2€€€€€€€€€ФФ
™ "2™/
-
dense$К!
dense€€€€€€€€€Ф©
?__inference_dense_layer_call_and_return_conditional_losses_4558f !4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Фd
™ "*Ґ'
 К
0€€€€€€€€€Ф
Ъ Б
$__inference_dense_layer_call_fn_4567Y !4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Фd
™ "К€€€€€€€€€Фт
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_4438©uҐr
[ҐX
VЪS
'К$
inputs/0€€€€€€€€€Фd
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp "*Ґ'
 К
0€€€€€€€€€Фd
Ъ т
D__inference_gat_conv_1_layer_call_and_return_conditional_losses_4499©uҐr
[ҐX
VЪS
'К$
inputs/0€€€€€€€€€Фd
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp"*Ґ'
 К
0€€€€€€€€€Фd
Ъ  
)__inference_gat_conv_1_layer_call_fn_4513ЬuҐr
[ҐX
VЪS
'К$
inputs/0€€€€€€€€€Фd
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp "К€€€€€€€€€Фd 
)__inference_gat_conv_1_layer_call_fn_4527ЬuҐr
[ҐX
VЪS
'К$
inputs/0€€€€€€€€€Фd
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp"К€€€€€€€€€Фdс
B__inference_gat_conv_layer_call_and_return_conditional_losses_4288™vҐs
\ҐY
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp "*Ґ'
 К
0€€€€€€€€€Фd
Ъ с
B__inference_gat_conv_layer_call_and_return_conditional_losses_4349™vҐs
\ҐY
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp"*Ґ'
 К
0€€€€€€€€€Фd
Ъ …
'__inference_gat_conv_layer_call_fn_4363ЭvҐs
\ҐY
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp "К€€€€€€€€€Фd…
'__inference_gat_conv_layer_call_fn_4377ЭvҐs
\ҐY
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
™

trainingp"К€€€€€€€€€Фdк
?__inference_model_layer_call_and_return_conditional_losses_3825¶
 !lҐi
bҐ_
UЪR
'К$
input_1€€€€€€€€€ФЩ
'К$
input_2€€€€€€€€€ФФ
p 

 
™ "*Ґ'
 К
0€€€€€€€€€Ф
Ъ к
?__inference_model_layer_call_and_return_conditional_losses_3853¶
 !lҐi
bҐ_
UЪR
'К$
input_1€€€€€€€€€ФЩ
'К$
input_2€€€€€€€€€ФФ
p

 
™ "*Ґ'
 К
0€€€€€€€€€Ф
Ъ м
?__inference_model_layer_call_and_return_conditional_losses_4031®
 !nҐk
dҐa
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
p 

 
™ "*Ґ'
 К
0€€€€€€€€€Ф
Ъ м
?__inference_model_layer_call_and_return_conditional_losses_4175®
 !nҐk
dҐa
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
p

 
™ "*Ґ'
 К
0€€€€€€€€€Ф
Ъ ¬
$__inference_model_layer_call_fn_3501Щ
 !lҐi
bҐ_
UЪR
'К$
input_1€€€€€€€€€ФЩ
'К$
input_2€€€€€€€€€ФФ
p 

 
™ "К€€€€€€€€€Ф¬
$__inference_model_layer_call_fn_3797Щ
 !lҐi
bҐ_
UЪR
'К$
input_1€€€€€€€€€ФЩ
'К$
input_2€€€€€€€€€ФФ
p

 
™ "К€€€€€€€€€Фƒ
$__inference_model_layer_call_fn_4201Ы
 !nҐk
dҐa
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
p 

 
™ "К€€€€€€€€€Фƒ
$__inference_model_layer_call_fn_4227Ы
 !nҐk
dҐa
WЪT
(К%
inputs/0€€€€€€€€€ФЩ
(К%
inputs/1€€€€€€€€€ФФ
p

 
™ "К€€€€€€€€€Фё
"__inference_signature_wrapper_3887Ј
 !uҐr
Ґ 
k™h
2
input_1'К$
input_1€€€€€€€€€ФЩ
2
input_2'К$
input_2€€€€€€€€€ФФ"2™/
-
dense$К!
dense€€€€€€€€€Ф