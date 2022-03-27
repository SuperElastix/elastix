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
#ifndef xoutrow_h
#define xoutrow_h

#include "xoutbase.h"
#include "xoutcell.h"

#include <memory> // For unique_ptr.
#include <sstream>

namespace xoutlibrary
{

/**
 * \class xoutrow
 * \brief The xoutrow class can easily generate tables.
 *
 * The xoutrow class is used in elastix for printing the registration
 * information, such as metric value, gradient information, etc. You
 * can fill in all this information, and only after calling
 * WriteBufferedData() the entire row is printed to the desired outputs.
 *
 * \ingroup xout
 */

class xoutrow : public xoutbase
{
public:
  using Self = xoutrow;
  using Superclass = xoutbase;

  /** Constructor */
  xoutrow() = default;

  /** Destructor */
  ~xoutrow() override = default;

  /** Write the buffered cell data in a row to the outputs,
   * separated by tabs.
   */
  void
  WriteBufferedData() override;

  /** Writes the names of the target cells to the outputs.
   */
  virtual void
  WriteHeaders();

  /** This method adds a new xoutcell to the map of Targets. */
  int
  AddNewTargetCell(const char * name);

  /** This method removes an xoutcell to the map of Targets. */
  int
  RemoveTargetCell(const char * name) override;

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).
   * In addition to the behaviour of the Superclass's methods, these functions
   * set the outputs of the TargetCells as well.
   */
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

protected:
  /** Method to set all targets at once. The outputs of these targets
   * are not set automatically, so make sure to do it yourself.
   */
  void
  SetXTargetCells(const XStreamMapType & cellmap) override;

private:
  std::map<std::string, std::unique_ptr<xoutbase>> m_CellMap;
};

} // end namespace xoutlibrary

#endif // end #ifndef xoutrow_h
