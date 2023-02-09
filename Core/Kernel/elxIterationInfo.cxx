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


#include "elxIterationInfo.h" // Its own header
#include "elxlog.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>


namespace elastix
{

std::ostream & IterationInfo::operator[](const char * const cellName)
{
  return m_CellMap[cellName];
}


void
IterationInfo::AddNewTargetCell(const char * const cellName)
{
  m_CellMap[cellName] = {};
}


void
IterationInfo::RemoveTargetCell(const char * const cellName)
{
  m_CellMap.erase(cellName);
}

void
IterationInfo::WriteHeaders() const
{
  std::string headers;
  const auto  begin = m_CellMap.cbegin();
  const auto  end = m_CellMap.cend();

  if (begin != end)
  {
    headers = begin->first;

    for (auto it = std::next(begin); it != end; ++it)
    {
      headers.push_back('\t');
      headers += it->first;
    }
  }
  log::info(headers);

  if (m_OutputFile)
  {
    *m_OutputFile << headers << std::endl;
  }
}


void
IterationInfo::WriteBufferedData()
{
  std::string data;
  const auto  begin = m_CellMap.cbegin();
  const auto  end = m_CellMap.cend();

  if (begin != end)
  {
    data = begin->second.str();

    for (auto it = std::next(begin); it != end; ++it)
    {
      data.push_back('\t');
      data += it->second.str();
    }
  }
  log::info(data);

  if (m_OutputFile)
  {
    *m_OutputFile << data << std::endl;
  }

  for (auto & cell : m_CellMap)
  {
    cell.second.str("");
  }
}


void
IterationInfo::RemoveOutputFile()
{
  m_OutputFile = nullptr;
}


void
IterationInfo::SetOutputFile(std::ofstream & outputFile)
{
  m_OutputFile = &outputFile;
}

} // namespace elastix
