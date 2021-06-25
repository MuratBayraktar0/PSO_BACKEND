from flask import  Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/pso', methods=['GET', 'POST'])
def run():
    import PSO
    return jsonify(subsetPerformance=PSO.subset_performance,oldFeature=int(PSO.oldFeature),newFeature=int(PSO.newFeature), newData=PSO.newData)
    #return PSO.subset_performance

if __name__ == "__main__":
    app.run()