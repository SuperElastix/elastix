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
#ifndef xoutcell_h
#define xoutcell_h

#include "xoutbase.h"
#include <sstream>

namespace xoutlibrary
{
using namespace std;

/**
 * \class xoutcell
 * \brief Stores the input in a string stream.
 *
 * The xoutcell class is used in the xoutrow class. It stores
 * input for a cell in a row.
 *
 * \ingroup xout
 */

template <class charT, class traits = char_traits<charT>>
class xoutcell : public xoutbase<charT, traits>
{
public:
  /** Typdef's. */
  typedef xoutcell                Self;
  typedef xoutbase<charT, traits> Superclass;

  /** Constructors */
  xoutcell();

  /** Destructor */
  ~xoutcell() override = default;

  /** Write the buffered cell data to the outputs. */
  void
  WriteBufferedData(void) override;

private:
  typedef std::basic_ostringstream<charT, traits> InternalBufferType;

  InternalBufferType m_InternalBuffer;
};

} // end namespace xoutlibrary

#include "xoutcell.hxx"

#endif // end #ifndef xoutcell_h
