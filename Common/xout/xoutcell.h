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

/**
 * \class xoutcell
 * \brief Stores the input in a string stream.
 *
 * The xoutcell class is used in the xoutrow class. It stores
 * input for a cell in a row.
 *
 * \ingroup xout
 */

class xoutcell : public xoutbase
{
public:
  /** Typdef's. */
  using Self = xoutcell;
  using Superclass = xoutbase;

  /** Constructors */
  xoutcell();

  /** Destructor */
  ~xoutcell() override = default;

  /** Write the buffered cell data to the outputs. */
  void
  WriteBufferedData() override;

private:
  using InternalBufferType = std::ostringstream;

  InternalBufferType m_InternalBuffer;
};

} // end namespace xoutlibrary

#endif // end #ifndef xoutcell_h
