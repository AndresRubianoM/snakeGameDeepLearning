��
��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
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
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-0-g582c8d236cb8��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
�
(QNetwork/EncodingNetwork/dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*9
shared_name*(QNetwork/EncodingNetwork/dense_48/kernel
�
<QNetwork/EncodingNetwork/dense_48/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_48/kernel*
_output_shapes
:	�*
dtype0
�
&QNetwork/EncodingNetwork/dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&QNetwork/EncodingNetwork/dense_48/bias
�
:QNetwork/EncodingNetwork/dense_48/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_48/bias*
_output_shapes	
:�*
dtype0
�
(QNetwork/EncodingNetwork/dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(QNetwork/EncodingNetwork/dense_49/kernel
�
<QNetwork/EncodingNetwork/dense_49/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_49/kernel* 
_output_shapes
:
��*
dtype0
�
&QNetwork/EncodingNetwork/dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&QNetwork/EncodingNetwork/dense_49/bias
�
:QNetwork/EncodingNetwork/dense_49/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_49/bias*
_output_shapes	
:�*
dtype0
�
(QNetwork/EncodingNetwork/dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*9
shared_name*(QNetwork/EncodingNetwork/dense_50/kernel
�
<QNetwork/EncodingNetwork/dense_50/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_50/kernel* 
_output_shapes
:
��*
dtype0
�
&QNetwork/EncodingNetwork/dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*7
shared_name(&QNetwork/EncodingNetwork/dense_50/bias
�
:QNetwork/EncodingNetwork/dense_50/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_50/bias*
_output_shapes	
:�*
dtype0
�
QNetwork/dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameQNetwork/dense_51/kernel
�
,QNetwork/dense_51/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_51/kernel*
_output_shapes
:	�*
dtype0
�
QNetwork/dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameQNetwork/dense_51/bias
}
*QNetwork/dense_51/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_51/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
	3

4
5
6
7

0
 
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_48/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_48/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_49/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_49/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_50/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_50/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEQNetwork/dense_51/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEQNetwork/dense_51/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
t
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
n
_postprocessing_layers
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
8
0
1
2
	3

4
5
6
7
8
0
1
2
	3

4
5
6
7
�
regularization_losses
 layer_regularization_losses

!layers
	variables
trainable_variables
"non_trainable_variables
#layer_metrics
$metrics

%0
&1
'2
(3
 
*
0
1
2
	3

4
5
*
0
1
2
	3

4
5
�
regularization_losses
)layer_regularization_losses

*layers
	variables
trainable_variables
+non_trainable_variables
,layer_metrics
-metrics
 

0
1

0
1
�
regularization_losses
.layer_regularization_losses

/layers
	variables
trainable_variables
0non_trainable_variables
1layer_metrics
2metrics
 

0
1
 
 
 
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

kernel
bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
h

kernel
	bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h


kernel
bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
 

%0
&1
'2
(3
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
 
�
3regularization_losses
Clayer_regularization_losses

Dlayers
4	variables
5trainable_variables
Enon_trainable_variables
Flayer_metrics
Gmetrics
 

0
1

0
1
�
7regularization_losses
Hlayer_regularization_losses

Ilayers
8	variables
9trainable_variables
Jnon_trainable_variables
Klayer_metrics
Lmetrics
 

0
	1

0
	1
�
;regularization_losses
Mlayer_regularization_losses

Nlayers
<	variables
=trainable_variables
Onon_trainable_variables
Player_metrics
Qmetrics
 


0
1


0
1
�
?regularization_losses
Rlayer_regularization_losses

Slayers
@	variables
Atrainable_variables
Tnon_trainable_variables
Ulayer_metrics
Vmetrics
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
l
action_0/discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0/observationPlaceholder*'
_output_shapes
:���������*
dtype0	*
shape:���������
j
action_0/rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0/step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type(QNetwork/EncodingNetwork/dense_48/kernel&QNetwork/EncodingNetwork/dense_48/bias(QNetwork/EncodingNetwork/dense_49/kernel&QNetwork/EncodingNetwork/dense_49/bias(QNetwork/EncodingNetwork/dense_50/kernel&QNetwork/EncodingNetwork/dense_50/biasQNetwork/dense_51/kernelQNetwork/dense_51/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_130884971
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_130884983
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_130885005
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_130884998
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_48/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_48/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_49/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_49/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_50/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_50/bias/Read/ReadVariableOp,QNetwork/dense_51/kernel/Read/ReadVariableOp*QNetwork/dense_51/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_save_130885241
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable(QNetwork/EncodingNetwork/dense_48/kernel&QNetwork/EncodingNetwork/dense_48/bias(QNetwork/EncodingNetwork/dense_49/kernel&QNetwork/EncodingNetwork/dense_49/bias(QNetwork/EncodingNetwork/dense_50/kernel&QNetwork/EncodingNetwork/dense_50/biasQNetwork/dense_51/kernelQNetwork/dense_51/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference__traced_restore_130885278��
�
g
-__inference_function_with_signature_130884990
unknown
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference_<lambda>_1308847372
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
�

�
-__inference_function_with_signature_130884945
	step_type

reward
discount
observation	
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *4
f/R-
+__inference_polymorphic_action_fn_1308849262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation
�
a
'__inference_signature_wrapper_130884998
unknown
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1308849902
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
�
9
'__inference_get_initial_state_130884977

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
?
-__inference_function_with_signature_130884978

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_get_initial_state_1308849772
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
)
'__inference_signature_wrapper_130885005�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1308850012
PartitionedCall*
_input_shapes 
5
 
__inference_<lambda>_130884740*
_input_shapes 
�e
�
+__inference_polymorphic_action_fn_130885071
	step_type

reward
discount
observation	D
@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource4
0qnetwork_dense_51_matmul_readvariableop_resource5
1qnetwork_dense_51_biasadd_readvariableop_resource
identity��8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�(QNetwork/dense_51/BiasAdd/ReadVariableOp�'QNetwork/dense_51/MatMul/ReadVariableOp�
)QNetwork/EncodingNetwork/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2+
)QNetwork/EncodingNetwork/flatten_12/Const�
+QNetwork/EncodingNetwork/flatten_12/ReshapeReshapeobservation2QNetwork/EncodingNetwork/flatten_12/Const:output:0*
T0	*'
_output_shapes
:���������2-
+QNetwork/EncodingNetwork/flatten_12/Reshape�
&QNetwork/EncodingNetwork/dense_48/CastCast4QNetwork/EncodingNetwork/flatten_12/Reshape:output:0*

DstT0*

SrcT0	*'
_output_shapes
:���������2(
&QNetwork/EncodingNetwork/dense_48/Cast�
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_48/MatMulMatMul*QNetwork/EncodingNetwork/dense_48/Cast:y:0?QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_48/MatMul�
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_48/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_48/MatMul:product:0@QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_48/BiasAdd�
&QNetwork/EncodingNetwork/dense_48/ReluRelu2QNetwork/EncodingNetwork/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_48/Relu�
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_49/MatMulMatMul4QNetwork/EncodingNetwork/dense_48/Relu:activations:0?QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_49/MatMul�
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_49/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_49/MatMul:product:0@QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_49/BiasAdd�
&QNetwork/EncodingNetwork/dense_49/ReluRelu2QNetwork/EncodingNetwork/dense_49/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_49/Relu�
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_50/MatMulMatMul4QNetwork/EncodingNetwork/dense_49/Relu:activations:0?QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_50/MatMul�
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_50/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_50/MatMul:product:0@QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_50/BiasAdd�
&QNetwork/EncodingNetwork/dense_50/ReluRelu2QNetwork/EncodingNetwork/dense_50/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_50/Relu�
'QNetwork/dense_51/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_51_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'QNetwork/dense_51/MatMul/ReadVariableOp�
QNetwork/dense_51/MatMulMatMul4QNetwork/EncodingNetwork/dense_50/Relu:activations:0/QNetwork/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/MatMul�
(QNetwork/dense_51/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_51/BiasAdd/ReadVariableOp�
QNetwork/dense_51/BiasAddBiasAdd"QNetwork/dense_51/MatMul:product:00QNetwork/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_51/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x�
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1�
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value�
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp)^QNetwork/dense_51/BiasAdd/ReadVariableOp(^QNetwork/dense_51/MatMul/ReadVariableOp*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������::::::::2t
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp2T
(QNetwork/dense_51/BiasAdd/ReadVariableOp(QNetwork/dense_51/BiasAdd/ReadVariableOp2R
'QNetwork/dense_51/MatMul/ReadVariableOp'QNetwork/dense_51/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�
/
-__inference_function_with_signature_130885001�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference_<lambda>_1308847402
PartitionedCall*
_input_shapes 
�
9
'__inference_signature_wrapper_130884983

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1308849782
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�f
�
+__inference_polymorphic_action_fn_130885137
time_step_step_type
time_step_reward
time_step_discount
time_step_observation	D
@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource4
0qnetwork_dense_51_matmul_readvariableop_resource5
1qnetwork_dense_51_biasadd_readvariableop_resource
identity��8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�(QNetwork/dense_51/BiasAdd/ReadVariableOp�'QNetwork/dense_51/MatMul/ReadVariableOp�
)QNetwork/EncodingNetwork/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2+
)QNetwork/EncodingNetwork/flatten_12/Const�
+QNetwork/EncodingNetwork/flatten_12/ReshapeReshapetime_step_observation2QNetwork/EncodingNetwork/flatten_12/Const:output:0*
T0	*'
_output_shapes
:���������2-
+QNetwork/EncodingNetwork/flatten_12/Reshape�
&QNetwork/EncodingNetwork/dense_48/CastCast4QNetwork/EncodingNetwork/flatten_12/Reshape:output:0*

DstT0*

SrcT0	*'
_output_shapes
:���������2(
&QNetwork/EncodingNetwork/dense_48/Cast�
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_48/MatMulMatMul*QNetwork/EncodingNetwork/dense_48/Cast:y:0?QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_48/MatMul�
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_48/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_48/MatMul:product:0@QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_48/BiasAdd�
&QNetwork/EncodingNetwork/dense_48/ReluRelu2QNetwork/EncodingNetwork/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_48/Relu�
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_49/MatMulMatMul4QNetwork/EncodingNetwork/dense_48/Relu:activations:0?QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_49/MatMul�
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_49/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_49/MatMul:product:0@QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_49/BiasAdd�
&QNetwork/EncodingNetwork/dense_49/ReluRelu2QNetwork/EncodingNetwork/dense_49/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_49/Relu�
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_50/MatMulMatMul4QNetwork/EncodingNetwork/dense_49/Relu:activations:0?QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_50/MatMul�
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_50/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_50/MatMul:product:0@QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_50/BiasAdd�
&QNetwork/EncodingNetwork/dense_50/ReluRelu2QNetwork/EncodingNetwork/dense_50/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_50/Relu�
'QNetwork/dense_51/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_51_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'QNetwork/dense_51/MatMul/ReadVariableOp�
QNetwork/dense_51/MatMulMatMul4QNetwork/EncodingNetwork/dense_50/Relu:activations:0/QNetwork/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/MatMul�
(QNetwork/dense_51/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_51/BiasAdd/ReadVariableOp�
QNetwork/dense_51/BiasAddBiasAdd"QNetwork/dense_51/MatMul:product:00QNetwork/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_51/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x�
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1�
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value�
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp)^QNetwork/dense_51/BiasAdd/ReadVariableOp(^QNetwork/dense_51/MatMul/ReadVariableOp*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������::::::::2t
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp2T
(QNetwork/dense_51/BiasAdd/ReadVariableOp(QNetwork/dense_51/BiasAdd/ReadVariableOp2R
'QNetwork/dense_51/MatMul/ReadVariableOp'QNetwork/dense_51/MatMul/ReadVariableOp:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:���������
/
_user_specified_nametime_step/observation
�e
�
+__inference_polymorphic_action_fn_130884926
	time_step
time_step_1
time_step_2
time_step_3	D
@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource4
0qnetwork_dense_51_matmul_readvariableop_resource5
1qnetwork_dense_51_biasadd_readvariableop_resource
identity��8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�(QNetwork/dense_51/BiasAdd/ReadVariableOp�'QNetwork/dense_51/MatMul/ReadVariableOp�
)QNetwork/EncodingNetwork/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2+
)QNetwork/EncodingNetwork/flatten_12/Const�
+QNetwork/EncodingNetwork/flatten_12/ReshapeReshapetime_step_32QNetwork/EncodingNetwork/flatten_12/Const:output:0*
T0	*'
_output_shapes
:���������2-
+QNetwork/EncodingNetwork/flatten_12/Reshape�
&QNetwork/EncodingNetwork/dense_48/CastCast4QNetwork/EncodingNetwork/flatten_12/Reshape:output:0*

DstT0*

SrcT0	*'
_output_shapes
:���������2(
&QNetwork/EncodingNetwork/dense_48/Cast�
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_48/MatMulMatMul*QNetwork/EncodingNetwork/dense_48/Cast:y:0?QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_48/MatMul�
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_48/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_48/MatMul:product:0@QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_48/BiasAdd�
&QNetwork/EncodingNetwork/dense_48/ReluRelu2QNetwork/EncodingNetwork/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_48/Relu�
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_49/MatMulMatMul4QNetwork/EncodingNetwork/dense_48/Relu:activations:0?QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_49/MatMul�
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_49/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_49/MatMul:product:0@QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_49/BiasAdd�
&QNetwork/EncodingNetwork/dense_49/ReluRelu2QNetwork/EncodingNetwork/dense_49/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_49/Relu�
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_50/MatMulMatMul4QNetwork/EncodingNetwork/dense_49/Relu:activations:0?QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_50/MatMul�
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_50/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_50/MatMul:product:0@QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_50/BiasAdd�
&QNetwork/EncodingNetwork/dense_50/ReluRelu2QNetwork/EncodingNetwork/dense_50/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_50/Relu�
'QNetwork/dense_51/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_51_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'QNetwork/dense_51/MatMul/ReadVariableOp�
QNetwork/dense_51/MatMulMatMul4QNetwork/EncodingNetwork/dense_50/Relu:activations:0/QNetwork/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/MatMul�
(QNetwork/dense_51/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_51/BiasAdd/ReadVariableOp�
QNetwork/dense_51/BiasAddBiasAdd"QNetwork/dense_51/MatMul:product:00QNetwork/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_51/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x�
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1�
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value�
IdentityIdentityclip_by_value:z:09^QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp)^QNetwork/dense_51/BiasAdd/ReadVariableOp(^QNetwork/dense_51/MatMul/ReadVariableOp*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������::::::::2t
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp2T
(QNetwork/dense_51/BiasAdd/ReadVariableOp(QNetwork/dense_51/BiasAdd/ReadVariableOp2R
'QNetwork/dense_51/MatMul/ReadVariableOp'QNetwork/dense_51/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������
#
_user_specified_name	time_step
�
_
__inference_<lambda>_130884737
readvariableop_resource
identity	��ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
�

�
'__inference_signature_wrapper_130884971
discount
observation	

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1308849452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�+
�
%__inference__traced_restore_130885278
file_prefix
assignvariableop_variable?
;assignvariableop_1_qnetwork_encodingnetwork_dense_48_kernel=
9assignvariableop_2_qnetwork_encodingnetwork_dense_48_bias?
;assignvariableop_3_qnetwork_encodingnetwork_dense_49_kernel=
9assignvariableop_4_qnetwork_encodingnetwork_dense_49_bias?
;assignvariableop_5_qnetwork_encodingnetwork_dense_50_kernel=
9assignvariableop_6_qnetwork_encodingnetwork_dense_50_bias/
+assignvariableop_7_qnetwork_dense_51_kernel-
)assignvariableop_8_qnetwork_dense_51_bias
identity_10��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*�
value�B�
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp;assignvariableop_1_qnetwork_encodingnetwork_dense_48_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp9assignvariableop_2_qnetwork_encodingnetwork_dense_48_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp;assignvariableop_3_qnetwork_encodingnetwork_dense_49_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp9assignvariableop_4_qnetwork_encodingnetwork_dense_49_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp;assignvariableop_5_qnetwork_encodingnetwork_dense_50_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp9assignvariableop_6_qnetwork_encodingnetwork_dense_50_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_qnetwork_dense_51_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_qnetwork_dense_51_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_9�
Identity_10IdentityIdentity_9:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*
T0*
_output_shapes
: 2
Identity_10"#
identity_10Identity_10:output:0*9
_input_shapes(
&: :::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
9
'__inference_get_initial_state_130885186

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�H
�
1__inference_polymorphic_distribution_fn_130885183
	step_type

reward
discount
observation	D
@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resourceD
@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resourceE
Aqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource4
0qnetwork_dense_51_matmul_readvariableop_resource5
1qnetwork_dense_51_biasadd_readvariableop_resource
identity��8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�(QNetwork/dense_51/BiasAdd/ReadVariableOp�'QNetwork/dense_51/MatMul/ReadVariableOp�
)QNetwork/EncodingNetwork/flatten_12/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2+
)QNetwork/EncodingNetwork/flatten_12/Const�
+QNetwork/EncodingNetwork/flatten_12/ReshapeReshapeobservation2QNetwork/EncodingNetwork/flatten_12/Const:output:0*
T0	*'
_output_shapes
:���������2-
+QNetwork/EncodingNetwork/flatten_12/Reshape�
&QNetwork/EncodingNetwork/dense_48/CastCast4QNetwork/EncodingNetwork/flatten_12/Reshape:output:0*

DstT0*

SrcT0	*'
_output_shapes
:���������2(
&QNetwork/EncodingNetwork/dense_48/Cast�
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_48/MatMulMatMul*QNetwork/EncodingNetwork/dense_48/Cast:y:0?QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_48/MatMul�
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_48/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_48/MatMul:product:0@QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_48/BiasAdd�
&QNetwork/EncodingNetwork/dense_48/ReluRelu2QNetwork/EncodingNetwork/dense_48/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_48/Relu�
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_49_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_49/MatMulMatMul4QNetwork/EncodingNetwork/dense_48/Relu:activations:0?QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_49/MatMul�
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_49/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_49/MatMul:product:0@QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_49/BiasAdd�
&QNetwork/EncodingNetwork/dense_49/ReluRelu2QNetwork/EncodingNetwork/dense_49/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_49/Relu�
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_50_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype029
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp�
(QNetwork/EncodingNetwork/dense_50/MatMulMatMul4QNetwork/EncodingNetwork/dense_49/Relu:activations:0?QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(QNetwork/EncodingNetwork/dense_50/MatMul�
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp�
)QNetwork/EncodingNetwork/dense_50/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_50/MatMul:product:0@QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)QNetwork/EncodingNetwork/dense_50/BiasAdd�
&QNetwork/EncodingNetwork/dense_50/ReluRelu2QNetwork/EncodingNetwork/dense_50/BiasAdd:output:0*
T0*(
_output_shapes
:����������2(
&QNetwork/EncodingNetwork/dense_50/Relu�
'QNetwork/dense_51/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_51_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'QNetwork/dense_51/MatMul/ReadVariableOp�
QNetwork/dense_51/MatMulMatMul4QNetwork/EncodingNetwork/dense_50/Relu:activations:0/QNetwork/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/MatMul�
(QNetwork/dense_51/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_51/BiasAdd/ReadVariableOp�
QNetwork/dense_51/BiasAddBiasAdd"QNetwork/dense_51/MatMul:product:00QNetwork/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
QNetwork/dense_51/BiasAdd�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_51/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtol�
IdentityIdentityCategorical_1/mode/Cast:y:09^QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp9^QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp)^QNetwork/dense_51/BiasAdd/ReadVariableOp(^QNetwork/dense_51/MatMul/ReadVariableOp*
T0*#
_output_shapes
:���������2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������::::::::2t
8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_48/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_48/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_49/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_49/MatMul/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_50/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_50/MatMul/ReadVariableOp2T
(QNetwork/dense_51/BiasAdd/ReadVariableOp(QNetwork/dense_51/BiasAdd/ReadVariableOp2R
'QNetwork/dense_51/MatMul/ReadVariableOp'QNetwork/dense_51/MatMul/ReadVariableOp:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������
%
_user_specified_nameobservation
�"
�
"__inference__traced_save_130885241
file_prefix'
#savev2_variable_read_readvariableop	G
Csavev2_qnetwork_encodingnetwork_dense_48_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_48_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_49_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_49_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_50_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_50_bias_read_readvariableop7
3savev2_qnetwork_dense_51_kernel_read_readvariableop5
1savev2_qnetwork_dense_51_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*�
value�B�
B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_48_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_48_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_49_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_49_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_50_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_50_bias_read_readvariableop3savev2_qnetwork_dense_51_kernel_read_readvariableop1savev2_qnetwork_dense_51_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*b
_input_shapesQ
O: : :	�:�:
��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 	

_output_shapes
::


_output_shapes
: "�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0/discount:0���������
>
0/observation-
action_0/observation:0	���������
0
0/reward$
action_0/reward:0���������
6
0/step_type'
action_0/step_type:0���������6
action,
StatefulPartitionedCall:0���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:��
�

train_step
metadata
model_variables
_all_assets

signatures

Waction
Xdistribution
Yget_initial_state
Zget_metadata
[get_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
Y
0
1
2
	3

4
5
6
7"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

\action
]get_initial_state
^get_train_step
_get_metadata"
signature_map
;:9	�2(QNetwork/EncodingNetwork/dense_48/kernel
5:3�2&QNetwork/EncodingNetwork/dense_48/bias
<::
��2(QNetwork/EncodingNetwork/dense_49/kernel
5:3�2&QNetwork/EncodingNetwork/dense_49/bias
<::
��2(QNetwork/EncodingNetwork/dense_50/kernel
5:3�2&QNetwork/EncodingNetwork/dense_50/bias
+:)	�2QNetwork/dense_51/kernel
$:"2QNetwork/dense_51/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
�
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
_postprocessing_layers
regularization_losses
	variables
trainable_variables
	keras_api
b__call__
*c&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
 "
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
�
regularization_losses
 layer_regularization_losses

!layers
	variables
trainable_variables
"non_trainable_variables
#layer_metrics
$metrics
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
<
%0
&1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
	3

4
5"
trackable_list_wrapper
J
0
1
2
	3

4
5"
trackable_list_wrapper
�
regularization_losses
)layer_regularization_losses

*layers
	variables
trainable_variables
+non_trainable_variables
,layer_metrics
-metrics
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
.layer_regularization_losses

/layers
	variables
trainable_variables
0non_trainable_variables
1layer_metrics
2metrics
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
3regularization_losses
4	variables
5trainable_variables
6	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_12", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
h__call__
*i&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 17}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 17]}}
�

kernel
	bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
j__call__
*k&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_49", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
�


kernel
bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
l__call__
*m&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 128]}}
 "
trackable_list_wrapper
<
%0
&1
'2
(3"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
3regularization_losses
Clayer_regularization_losses

Dlayers
4	variables
5trainable_variables
Enon_trainable_variables
Flayer_metrics
Gmetrics
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
7regularization_losses
Hlayer_regularization_losses

Ilayers
8	variables
9trainable_variables
Jnon_trainable_variables
Klayer_metrics
Lmetrics
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
�
;regularization_losses
Mlayer_regularization_losses

Nlayers
<	variables
=trainable_variables
Onon_trainable_variables
Player_metrics
Qmetrics
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
�
?regularization_losses
Rlayer_regularization_losses

Slayers
@	variables
Atrainable_variables
Tnon_trainable_variables
Ulayer_metrics
Vmetrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
�2�
+__inference_polymorphic_action_fn_130885137
+__inference_polymorphic_action_fn_130885071�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_polymorphic_distribution_fn_130885183�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_get_initial_state_130885186�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_130884740"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_130884737"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_130884971
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_130884983
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_130884998"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_signature_wrapper_130885005"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 =
__inference_<lambda>_130884737�

� 
� "� 	6
__inference_<lambda>_130884740�

� 
� "� T
'__inference_get_initial_state_130885186)"�
�
�

batch_size 
� "� �
+__inference_polymorphic_action_fn_130885071�	
���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������	
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
+__inference_polymorphic_action_fn_130885137�	
���
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������>
observation/�,
time_step/observation���������	
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
1__inference_polymorphic_distribution_fn_130885183�	
���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������	
� 
� "���

PolicyStep�
action�����Ã���
`
C�@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*�'
%
loc�
Identity���������
`�]

allow_nan_statsp


atol
 

namejDeterministic


rtol
 

validate_argsp _DistributionTypeSpec
state� 
info� �
'__inference_signature_wrapper_130884971�	
���
� 
���
.

0/discount �

0/discount���������
8
0/observation'�$
0/observation���������	
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������"+�(
&
action�
action���������b
'__inference_signature_wrapper_13088498370�-
� 
&�#
!

batch_size�

batch_size "� [
'__inference_signature_wrapper_1308849980�

� 
� "�

int64�
int64 	?
'__inference_signature_wrapper_130885005�

� 
� "� 