import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import yaml
from shutil import copy2
import sys
import json
import importlib

import http.server
import socketserver
import threading
from functools import partial
import cv2
import base64

# x: N by 4: r x y z
def attrswapaxis(x):
    y = x * torch.tensor([1, 1,-1,-1], dtype=x.dtype, device=x.device)
    return y
    
class TestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer
        self.start_path = os.path.abspath('pyjs3d')
        super().__init__(*args, **kwargs)
        
    def send_head(self):
        path = self.translate_path(self.path)
        f = None
        try:
            if not path.startswith(self.start_path):
                raise IOError
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            #self.send_error(404, "File not found")
            self.send_response(301)
            self.send_header("Location", "/pyjs3d/html/webgl_dualsdf_editor.html")
            self.end_headers()
            return None
        ctype = self.guess_type(path)
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f
        
    def do_POST(self):
        print('********HEADER********')
        #print(self.headers)
        length = int(self.headers.get_all('content-length')[0])
        print('********content-length********')
        print(self.headers.get_all('content-length'))
        print('********data_string********')
        data_string = self.rfile.read(length)
        #print(data_string)

        data = json.loads(data_string)
        #print(data)
        if self.path == '/get_attributes':
            exist_feature = None
            if 'feature' in data.keys():
                exist_feature = np.array(data['feature'], dtype=np.float32)
            attrs, feature = self.handle_get_new_shape(exist_feature)
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.flush_headers()
            self.wfile.write(json.dumps({'attrs': attrs.tolist(), 'feature': feature.tolist(), 'kld': self.trainer.stats_loss_kld}).encode())
        elif self.path == '/update_attributes':
            attrs = np.array(data['attrs'], dtype=np.float32)
            attrs_mod = np.array(data['modified_attrs'], dtype=np.float32)
            feature = np.array(data['feature'], dtype=np.float32)
            gamma = data['gamma']
            beta = data['beta']
            attrs_update, feature_update = self.handle_update_shape(attrs, attrs_mod, feature, gamma, beta)
            # Process the new primitives
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.flush_headers()
            self.wfile.write(json.dumps({'attrs': attrs_update.tolist(), 'feature': feature_update.tolist(), 'kld': self.trainer.stats_loss_kld}).encode())
        elif self.path == '/get_highres':
            print('get_highres')
            feature = np.array(data['feature'], dtype=np.float32)
            feature = torch.from_numpy(feature)
            
            rendered_img = self.trainer.render_express(feature)
            rendered_img_png = cv2.imencode('.png', rendered_img, [cv2.IMWRITE_PNG_COMPRESSION,3])[1]
            rendered_img_png_b64 = "data:image/png;base64, " + base64.b64encode(rendered_img_png).decode('ascii')
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.flush_headers()
            print(len(rendered_img_png_b64))
            self.wfile.write(json.dumps({'highres_png': rendered_img_png_b64}).encode())
        else:
            print('Unknow POST path')
    
    def handle_update_shape(self, attrs, attrs_mod, feature, gamma, beta):
        attrs_mod = torch.from_numpy(attrs_mod)
        attrs_mod[:,0] = torch.log(attrs_mod[:,0])
        attrs = torch.from_numpy(attrs)
        attrs[:,0] = torch.log(attrs[:,0])
        # Convert manipulation to objective function
        mask = (torch.abs(attrs_mod-attrs) > 0.001).float()
        def lossfun(x, feat):
            loss_kld = torch.mean(0.5 * torch.mean(feat**2, dim=-1))
            x = attrswapaxis(x)
            gt = attrs_mod
            mask_ = mask
            gt = gt.to(x.device)
            mask_ = mask_.to(x.device)
            loss = torch.mean(torch.abs(x - gt)*mask_) + gamma * (torch.clamp(loss_kld, beta, None) - beta)
            return loss
        feature, attrs = self.trainer.step_manip(torch.from_numpy(feature), lossfun)
        attrs[:,0] = torch.exp(attrs[:,0])
        attrs = attrswapaxis(attrs)
        return attrs.detach().cpu().numpy(), feature.detach().cpu().numpy()
        
    def handle_get_new_shape(self, existing_feature=None):
        if existing_feature is None:
            num_features = self.trainer.get_known_latent(None)
            feature = self.trainer.get_known_latent(np.random.choice(num_features)) # 1, 128
        else:
            feature = torch.from_numpy(existing_feature).to(self.trainer.device)
        attrs = self.trainer.prim_attr_net(feature)
        attrs = attrs.reshape( -1, 4) # 64, 256, 4, (r x y z)
        attrs[:,0] = torch.exp(attrs[:,0])
        attrs = attrswapaxis(attrs)
        return attrs.detach().cpu().numpy(), feature.detach().cpu().numpy()

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    pass

def run(server_class=http.server.HTTPServer, handler_class=http.server.BaseHTTPRequestHandler, trainer=None, port=1234):
    handler = partial(handler_class, trainer)
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler)
    httpd.trainer = trainer
    httpd.serve_forever()

def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='DualSDF Web Demo')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    # Resume:
    parser.add_argument('--pretrained', default=None, type=str,
                        help='pretrained model checkpoint')

    parser.add_argument('--port', default=1234, type=int)
    
    args = parser.parse_args()
    
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)

            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = dict2namespace(config)

    return args, config

def main(args, cfg):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)
   
    if args.pretrained is not None:
        trainer.resume_demo(args.pretrained)
    else:
        trainer.resume_demo(cfg.resume.dir)
        
    run(http.server.HTTPServer, TestHandler, trainer, port=args.port)

    
if __name__ == "__main__":
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main(args, cfg)
    
    
