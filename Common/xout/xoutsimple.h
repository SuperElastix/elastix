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
#ifndef xoutsimple_h
#define xoutsimple_h

#include "xoutbase.h"

namespace xoutlibrary
{

/**
 * \class xoutsimple
 * \brief xout class with only basic functionality.
 *
 * The xoutsimple class just immediately prints to the desired outputs.
 *
 * \ingroup xout
 */

class xoutsimple : public xoutbase
{
public:
  /** Typedef's.*/
  using Self = xoutsimple;
  using Superclass = xoutbase;

  /** Constructors */
  xoutsimple() = default;

  /** Destructor */
  ~xoutsimple() override = default;

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).  */
  int
  AddOutput(const char * name, std::ostream * output) override;

  int
  AddOutput(const char * name, Superclass * output) override;

  int
  RemoveOutput(const char * name) override;

  void
  SetOutputs(const CStreamMapType & outputmap) override;

  void
  SetOutputs(const XStreamMapType & outputmap) override;

  /** Get the output maps. */
  const CStreamMapType &
  GetCOutputs() override;

  const XStreamMapType &
  GetXOutputs() override;
};

} // end namespace xoutlibrary

#endif // end #ifndef xoutsimple_h
