#include <bicudo/io/file.hpp>
#include <fstream>

bicudo::result_t bicudo::read(
  bicudo::file_t<std::string> &file,
  bicudo::file_read_properties_t<std::string> &file_read_properties
) {
  file.tag = file_read_properties.tag;
  file.path = file_read_properties.path;
  
  std::ifstream fstream {std::ifstream(file.path)};
  if (!fstream.is_open()) {
    bicudo::loge("Failed to open file '", file.path, "' - unkonwn reason!");
    return bicudo::result::FAILED_TO_OPEN_FILE;
  }

  std::string line {};
  while (std::getline(fstream, line)) file.content.append(line); 

  return bicudo::result::OK;
}
