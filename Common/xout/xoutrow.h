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
using namespace std;

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

template <class charT, class traits = char_traits<charT>>
class xoutrow : public xoutbase<charT, traits>
{
public:
  typedef xoutrow                 Self;
  typedef xoutbase<charT, traits> Superclass;

  /** Typedefs of Superclass */
  using typename Superclass::ostream_type;
  using typename Superclass::CStreamMapType;
  using typename Superclass::XStreamMapType;


  /** Extra typedefs */
  typedef xoutcell<charT, traits> XOutCellType;

  /** Constructor */
  xoutrow() = default;

  /** Destructor */
  ~xoutrow() override = default;

  /** Write the buffered cell data in a row to the outputs,
   * separated by tabs.
   */
  void
  WriteBufferedData(void) override;

  /** Writes the names of the target cells to the outputs;
   * This method can also be executed by selecting the
   * "WriteHeaders" target: xout["WriteHeaders"]
   */
  virtual void
  WriteHeaders(void);

  /** This method adds an xoutcell to the map of Targets. */
  int
  AddTargetCell(const char * name) override;

  /** This method removes an xoutcell to the map of Targets. */
  int
  RemoveTargetCell(const char * name) override;

  /** Method to set all targets at once. The outputs of these targets
   * are not set automatically, so make sure to do it yourself.
   */
  void
  SetTargetCells(const XStreamMapType & cellmap) override;

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).
   * In addition to the behaviour of the Superclass's methods, these functions
   * set the outputs of the TargetCells as well.
   */
  int
  AddOutput(const char * name, ostream_type * output) override;

  int
  AddOutput(const char * name, Superclass * output) override;

  int
  RemoveOutput(const char * name) override;

  void
  SetOutputs(const CStreamMapType & outputmap) override;

  void
  SetOutputs(const XStreamMapType & outputmap) override;

protected:
  /** Returns a target cell.
   * Extension: if input = "WriteHeaders" it calls
   * this->WriteHeaders() and returns 'this'.
   */
  Superclass &
  SelectXCell(const char * name) override;

private:
  std::map<std::string, std::unique_ptr<xoutbase<charT, traits>>> m_CellMap;
};

} // end namespace xoutlibrary

#include "xoutrow.hxx"

#endif // end #ifndef xoutrow_h
