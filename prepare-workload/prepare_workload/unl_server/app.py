import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

unl_json = json.loads(Path("/unl.json").read_text())

class S(BaseHTTPRequestHandler):

    def do_GET(self):
        # json_data = {'token': 'qfrwefewrtweygds--fefef==wef'}
        json_to_pass = json.dumps(unl_json)
        self.send_response(code=200) #, message='here is your token')
        self.send_header(keyword='Content-type', value='application/json')
        self.end_headers()
        self.wfile.write(json_to_pass.encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, addr="0.0.0.0", port=80):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
