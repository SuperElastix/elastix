/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef elxIterationInfo_hxx
#define elxIterationInfo_hxx

#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace elastix
{

class IterationInfo
{
public:
  std::ostream & operator[](const char * const cellName);

  void
  AddNewTargetCell(const char * const cellName);

  void
  RemoveTargetCell(const char * const cellName);

  void
  WriteHeaders() const;

  void
  WriteBufferedData();

  void
  RemoveOutputFile();

  void
  SetOutputFile(std::ofstream & outputFile);

private:
  std::map<std::string, std::ostringstream> m_CellMap{};
  std::ofstream *                           m_OutputFile{};
};

} // namespace elastix

#endif
