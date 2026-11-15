import urllib.request, io
class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size, block=1<<20):
        self.url, self.size, self.pos, self.block, self.cache = url, size, 0, block, {}
        self.nreq = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else (self.pos+off if whence==1 else self.size+off); return self.pos
    def _blk(self, b):
        if b not in self.cache:
            s=b*self.block; e=min(self.size, s+self.block)-1
            import subprocess
            self.cache[b]=subprocess.run(['curl','-sSL','--fail','--max-time','120','-r',f'{s}-{e}',self.url],capture_output=True,check=True).stdout; self.nreq+=1
            if len(self.cache)>512: self.cache.pop(next(iter(self.cache)))
        return self.cache[b]
    def readinto(self, buf):
        n=len(buf); n=min(n, self.size-self.pos)
        if n<=0: return 0
        out=bytearray()
        while len(out)<n:
            p=self.pos+len(out); b=p//self.block; d=self._blk(b); o=p-b*self.block
            out+=d[o:o+n-len(out)]
        buf[:n]=out; self.pos+=n; return n
