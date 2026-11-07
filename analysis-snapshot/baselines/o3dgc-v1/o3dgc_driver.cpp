// o3dgc_driver.cpp (2026-11-01, VERSION o3dgc-v1): encode/decode one MeshSplatting mesh with Open3DGC (MPEG-4 Part 16 SC3DMC, TFAN).
// Input bundle (little-endian, written by o3dgc_codec_v1.py):
//   char[8] "O3DGCB01" | uint32 V | uint32 F | uint32 mode (0 = float SH, 1 = uint8 SH) | uint32 D (SH dims, 48)
//   float32 xyz[V*3] | int32 faces[F*3] | float32 sh[V*D]  (mode 0)  or  uint8 sh[V*D]  (mode 1)
// Output: <out.o3dgc> (the SC3DMC bitstream, its size is the byte count) and <out.decoded> with the same layout as the input,
// holding the DECODED positions (dequantised) and SH (float32 dequantised in mode 0, uint8 lossless in mode 1).
// usage: o3dgc_driver in.bundle out.o3dgc out.decoded QCOORD QATTR
// Attributes are split into groups of 8 dims (O3DGC_SC3DMC_MAX_DIM_ATTRIBUTES) -> 6 float (mode 0) or 6 int (mode 1) attributes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "o3dgcCommon.h"
#include "o3dgcVector.h"
#include "o3dgcBinaryStream.h"
#include "o3dgcIndexedFaceSet.h"
#include "o3dgcSC3DMCEncodeParams.h"
#include "o3dgcSC3DMCEncoder.h"
#include "o3dgcSC3DMCDecoder.h"
using namespace o3dgc;
typedef unsigned long Index;
// the library spells the payload decoder "DecodePlayload"; accept either spelling
template <class D, class I, class B> auto decode_payload(D& d, I& ifs, B& bs, int) -> decltype(d.DecodePlayload(ifs, bs)) { return d.DecodePlayload(ifs, bs); }
template <class D, class I, class B> auto decode_payload(D& d, I& ifs, B& bs, long) -> decltype(d.DecodePayload(ifs, bs)) { return d.DecodePayload(ifs, bs); }
static void die(const char* m) { fprintf(stderr, "FAILED: %s\n", m); exit(1); }
template <class T> static void rd(FILE* f, std::vector<T>& v, size_t n) { v.resize(n); if (n && fread(v.data(), sizeof(T), n, f) != n) die("short read"); }
int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s in.bundle out.o3dgc out.decoded QCOORD QATTR\n", argv[0]); return 1; }
    const int qcoord = atoi(argv[4]), qattr = atoi(argv[5]);
    FILE* f = fopen(argv[1], "rb"); if (!f) die("open bundle");
    char magic[8]; if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "O3DGCB01", 8)) die("bad magic");
    unsigned int V, F, mode, D; if (fread(&V, 4, 1, f) != 1 || fread(&F, 4, 1, f) != 1 || fread(&mode, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1) die("bad header");
    std::vector<float> xyz; rd(f, xyz, (size_t)V * 3);
    std::vector<int> faces32; rd(f, faces32, (size_t)F * 3);
    std::vector<float> shf; std::vector<unsigned char> sh8;
    if (mode == 0) rd(f, shf, (size_t)V * D); else rd(f, sh8, (size_t)V * D);
    fclose(f);
    const unsigned int G = 8, NG = (D + G - 1) / G;  // attribute groups of <= 8 dims
    std::vector<Index> faces((size_t)F * 3); for (size_t i = 0; i < faces.size(); ++i) faces[i] = (Index)faces32[i];
    std::vector<std::vector<float> > fattr(NG); std::vector<std::vector<long> > iattr(NG);
    for (unsigned int g = 0; g < NG; ++g) {
        unsigned int d0 = g * G, dim = (D - d0 < G) ? (D - d0) : G;
        if (mode == 0) { fattr[g].resize((size_t)V * dim); for (unsigned int v = 0; v < V; ++v) for (unsigned int k = 0; k < dim; ++k) fattr[g][(size_t)v * dim + k] = shf[(size_t)v * D + d0 + k]; }
        else { iattr[g].resize((size_t)V * dim); for (unsigned int v = 0; v < V; ++v) for (unsigned int k = 0; k < dim; ++k) iattr[g][(size_t)v * dim + k] = (long)sh8[(size_t)v * D + d0 + k]; }
    }
    IndexedFaceSet<Index> ifs;
    ifs.SetNCoord(V); ifs.SetCoord(xyz.data());
    ifs.SetNCoordIndex(F); ifs.SetCoordIndex(faces.data());
    SC3DMCEncodeParams params;
    params.SetCoordQuantBits(qcoord);
    params.SetCoordPredMode(O3DGC_SC3DMC_PARALLELOGRAM_PREDICTION);
    params.SetStreamType(O3DGC_STREAM_TYPE_BINARY);
    params.SetEncodeMode(O3DGC_SC3DMC_ENCODE_MODE_TFAN);
    if (mode == 0) {
        ifs.SetNumFloatAttributes(NG); params.SetNumFloatAttributes(NG);
        for (unsigned int g = 0; g < NG; ++g) {
            unsigned int dim = fattr[g].size() / V;
            ifs.SetNFloatAttribute(g, V); ifs.SetFloatAttributeDim(g, dim); ifs.SetFloatAttributeType(g, O3DGC_IFS_FLOAT_ATTRIBUTE_TYPE_UNKOWN); ifs.SetFloatAttribute(g, fattr[g].data());
            params.SetFloatAttributeQuantBits(g, qattr); params.SetFloatAttributePredMode(g, O3DGC_SC3DMC_PARALLELOGRAM_PREDICTION);
        }
    } else {
        ifs.SetNumIntAttributes(NG); params.SetNumIntAttributes(NG);
        for (unsigned int g = 0; g < NG; ++g) {
            unsigned int dim = iattr[g].size() / V;
            ifs.SetNIntAttribute(g, V); ifs.SetIntAttributeDim(g, dim); ifs.SetIntAttributeType(g, O3DGC_IFS_INT_ATTRIBUTE_TYPE_UNKOWN); ifs.SetIntAttribute(g, iattr[g].data());
            params.SetIntAttributePredMode(g, O3DGC_SC3DMC_DIFFERENTIAL_PREDICTION);
        }
    }
    ifs.ComputeMinMax(O3DGC_SC3DMC_MAX_ALL_DIMS);  // one range per attribute = max extent over its dims (coords: the bounding cube, same meaning as our q<b>)
    BinaryStream bstream((unsigned long)V * 64 + (unsigned long)F * 16 + 4096);
    SC3DMCEncoder<Index> encoder;
    if (encoder.Encode(params, ifs, bstream) != O3DGC_OK) die("Encode");
    FILE* fo = fopen(argv[2], "wb"); if (!fo) die("open out.o3dgc");
    fwrite(bstream.GetBuffer(), 1, bstream.GetSize(), fo); fclose(fo);
    printf("O3DGC encoded V %u F %u mode %u D %u qcoord %d qattr %d -> %lu bytes\n", V, F, mode, D, qcoord, qattr, (unsigned long)bstream.GetSize());
    // ---- decode from the same stream
    IndexedFaceSet<Index> ifs2; SC3DMCDecoder<Index> decoder;
    if (decoder.DecodeHeader(ifs2, bstream) != O3DGC_OK) die("DecodeHeader");
    const unsigned int V2 = ifs2.GetNCoord(), F2 = ifs2.GetNCoordIndex();
    std::vector<float> xyz2((size_t)V2 * 3); std::vector<Index> faces2((size_t)F2 * 3);
    ifs2.SetCoord(xyz2.data()); ifs2.SetCoordIndex(faces2.data());
    std::vector<std::vector<float> > fattr2(ifs2.GetNumFloatAttributes()); std::vector<std::vector<long> > iattr2(ifs2.GetNumIntAttributes());
    for (unsigned int g = 0; g < ifs2.GetNumFloatAttributes(); ++g) { fattr2[g].resize((size_t)ifs2.GetNFloatAttribute(g) * ifs2.GetFloatAttributeDim(g)); ifs2.SetFloatAttribute(g, fattr2[g].data()); }
    for (unsigned int g = 0; g < ifs2.GetNumIntAttributes(); ++g) { iattr2[g].resize((size_t)ifs2.GetNIntAttribute(g) * ifs2.GetIntAttributeDim(g)); ifs2.SetIntAttribute(g, iattr2[g].data()); }
    if (decode_payload(decoder, ifs2, bstream, 0) != O3DGC_OK) die("DecodePayload");
    if (V2 != V || F2 != F) fprintf(stderr, "note: decoded V %u F %u (input %u %u)\n", V2, F2, V, F);
    FILE* fd = fopen(argv[3], "wb"); if (!fd) die("open out.decoded");
    fwrite("O3DGCB01", 1, 8, fd); fwrite(&V2, 4, 1, fd); fwrite(&F2, 4, 1, fd); fwrite(&mode, 4, 1, fd); fwrite(&D, 4, 1, fd);
    fwrite(xyz2.data(), 4, (size_t)V2 * 3, fd);
    std::vector<int> f32((size_t)F2 * 3); for (size_t i = 0; i < f32.size(); ++i) f32[i] = (int)faces2[i]; fwrite(f32.data(), 4, f32.size(), fd);
    if (mode == 0) { std::vector<float> out((size_t)V2 * D, 0.f); for (unsigned int g = 0; g < fattr2.size(); ++g) { unsigned int dim = ifs2.GetFloatAttributeDim(g); for (unsigned int v = 0; v < V2; ++v) for (unsigned int k = 0; k < dim; ++k) out[(size_t)v * D + g * G + k] = fattr2[g][(size_t)v * dim + k]; } fwrite(out.data(), 4, out.size(), fd); }
    else { std::vector<unsigned char> out((size_t)V2 * D, 0); for (unsigned int g = 0; g < iattr2.size(); ++g) { unsigned int dim = ifs2.GetIntAttributeDim(g); for (unsigned int v = 0; v < V2; ++v) for (unsigned int k = 0; k < dim; ++k) out[(size_t)v * D + g * G + k] = (unsigned char)iattr2[g][(size_t)v * dim + k]; } fwrite(out.data(), 1, out.size(), fd); }
    fclose(fd);
    printf("O3DGC decoded V %u F %u -> %s\n", V2, F2, argv[3]);
    return 0;
}
