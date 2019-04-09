from tensorflow import Session, Graph, GraphDef, import_graph_def, get_default_graph
from tensorflow.gfile import GFile
from tensorflow.saved_model.builder import SavedModelBuilder
from tensorflow.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.tag_constants import SERVING

export_dir = './inception_saved_model/1'
graph_pb_file = 'tensorflow_inception_graph.pb'

builder = SavedModelBuilder(export_dir)

with GFile(graph_pb_file, "rb") as f:
    graph_def = GraphDef()
    graph_def.ParseFromString(f.read())

signatures = {}

with Session(graph=Graph()) as sess:
     import_graph_def(graph_def, name="")
     g = get_default_graph()
     input_tensor = g.get_tensor_by_name("input:0")
     output_tensor = g.get_tensor_by_name("output:0")
     signatures[DEFAULT_SERVING_SIGNATURE_DEF_KEY] = predict_signature_def({"input": input_tensor}, {"output": output_tensor})
     builder.add_meta_graph_and_variables(sess, [SERVING], signature_def_map=signatures)

builder.save()
