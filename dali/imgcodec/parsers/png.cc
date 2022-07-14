// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dali/core/byte_io.h"
#include "dali/imgcodec/parsers/png.h"

namespace dali {
namespace imgcodec {

// https://www.w3.org/TR/2003/REC-PNG-20031110

enum ColorType : uint8_t {
  PNG_COLOR_TYPE_GRAY       = 0,
  PNG_COLOR_TYPE_RGB        = 2,
  PNG_COLOR_TYPE_PALETTE    = 3,
  PNG_COLOR_TYPE_GRAY_ALPHA = 4,
  PNG_COLOR_TYPE_RGBA       = 6
};

struct IhdrChunk {
  uint32_t width;
  uint32_t height;
  uint8_t color_type;
  // Some fields were ommited.

  int GetNumberOfChannels() {
    switch (color_type) {
      case PNG_COLOR_TYPE_GRAY:
      case PNG_COLOR_TYPE_GRAY_ALPHA:
        return 1;
      case PNG_COLOR_TYPE_RGB:
      case PNG_COLOR_TYPE_PALETTE:  // 1 byte but it's converted to 3-channel BGR by OpenCV
        return 3;
      case PNG_COLOR_TYPE_RGBA:
        return 4;
      default:
        DALI_FAIL(make_string("color type not supported: ", color_type));
    }
  }
};

using chunk_type_field_t = std::array<uint8_t, 4>;
static constexpr chunk_type_field_t expected_ihdr_type_field = {73, 72, 68, 82};

// Expects the read pointer in the stream to point to the beginning of a chunk.
static IhdrChunk ReadIhdrChunk(InputStream& stream) {
  // IHDR Chunk:
  //  IHDR chunk length(4 bytes): 0x00 0x00 0x00 0x0D
  //  IHDR chunk type(Identifies chunk type to be IHDR): 0x49 0x48 0x44 0x52
  //  Image width in pixels(variable 4): xx xx xx xx
  //  Image height in pixels(variable 4): xx xx xx xx
  //  Flags in the chunk(variable 5 bytes): xx xx xx xx xx
  //  CRC checksum(variable 4 bytes): xx xx xx xx

  uint32_t length = ReadValueBE<uint32_t>(stream);
  DALI_ENFORCE(length == 4 + 4 + 5);

  chunk_type_field_t chunk_type;
  stream.ReadAll<uint8_t>(chunk_type.data(), chunk_type.size());
  DALI_ENFORCE(chunk_type == expected_ihdr_type_field);

  IhdrChunk chunk;
  chunk.width = ReadValueBE<uint32_t>(stream);
  chunk.height = ReadValueBE<uint32_t>(stream);
  stream.Skip(1);  // Skip the bit depth info.
  chunk.color_type = ReadValueBE<uint8_t>(stream);
  stream.Skip(3 + 4);  // Skip the other fields and the CRC checksum.

  return chunk;
}

using png_signature_t = std::array<uint8_t, 8>;
static constexpr png_signature_t expected_signature = {137, 80, 78, 71, 13, 10, 26, 10};

ImageInfo PngParser::GetInfo(ImageSource *encoded) const {
  auto stream = encoded->Open();

  stream->Skip(expected_signature.size());  // Skip the PNG signature.
  auto ihdr = ReadIhdrChunk(*stream);  // First chunk is required to be IHDR.

  ImageInfo info;
  info.shape = {
    ihdr.height,
    ihdr.width,
    ihdr.GetNumberOfChannels()
  };
  return info;
}

bool PngParser::CanParse(ImageSource *encoded) const {
  png_signature_t buffer;
  if (ReadHeader(buffer.data(), encoded, buffer.size()) != expected_signature.size()) {
    return false;
  }

  return expected_signature == buffer;
}

}  // namespace imgcodec
}  // namespace dali
