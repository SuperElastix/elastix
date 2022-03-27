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
#ifndef xoutbase_h
#define xoutbase_h

#include <iostream>
#include <ostream>
#include <map>
#include <string>

namespace xoutlibrary
{

/**
 * \class xoutbase
 * \brief Base class for xout.
 *
 * An abstract base class, which defines the interface
 * for using xout.
 *
 * \ingroup xout
 */

class xoutbase
{
public:
  /** Typedef's.*/
  using Self = xoutbase;

  using CStreamMapType = std::map<std::string, std::ostream *>;
  using XStreamMapType = std::map<std::string, Self *>;
  using CStreamMapEntryType = CStreamMapType::value_type;
  using XStreamMapEntryType = XStreamMapType::value_type;

  /** Destructor */
  virtual ~xoutbase() = 0;

  /** The operator [] returns an x-cell */
  Self & operator[](const char * cellname);

  /**
   * the << operator. A templated member function, and some overloads.
   *
   * The overloads are required for manipulators, like std::endl.
   * (these manipulators in fact are global template functions,
   * and need to deduce their own template arguments)
   */

  /** template < class T >
      Self & operator<<(T &  _arg)
    {
      return this->SendToTargets(_arg);
    }*/

  template <class T>
  Self &
  operator<<(const T & _arg)
  {
    return this->SendToTargets(_arg);
  }


  Self &
  operator<<(std::ostream & (*pf)(std::ostream &))
  {
    return this->SendToTargets(pf);
  }


  Self &
  operator<<(std::ios & (*pf)(std::ios &))
  {
    return this->SendToTargets(pf);
  }


  Self &
  operator<<(std::ios_base & (*pf)(std::ios_base &))
  {
    return this->SendToTargets(pf);
  }


  virtual void
  WriteBufferedData();

  /**
   * Methods to Add and Remove target cells. They return 0 when successful.
   */
  virtual int
  AddTargetCell(const char * name, std::ostream * cell);

  virtual int
  AddTargetCell(const char * name, Self * cell);

  virtual int
  RemoveTargetCell(const char * name);

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).  */
  virtual int
  AddOutput(const char * name, std::ostream * output);

  virtual int
  AddOutput(const char * name, Self * output);

  virtual int
  RemoveOutput(const char * name);

  virtual void
  SetOutputs(const CStreamMapType & outputmap);

  virtual void
  SetOutputs(const XStreamMapType & outputmap);

  /** Get the output maps. */
  virtual const CStreamMapType &
  GetCOutputs();

  virtual const XStreamMapType &
  GetXOutputs();

protected:
  /** Default-constructor. Only to be used by its derived classes. */
  xoutbase() = default;

  void
  SetCTargetCells(const CStreamMapType & cellmap);

  virtual void
  SetXTargetCells(const XStreamMapType & cellmap);

  /** Maps that contain the outputs. */
  CStreamMapType m_COutputs;
  XStreamMapType m_XOutputs;

  /** Maps that contain the target cells. The << operator passes its
   * input to these maps. */
  CStreamMapType m_CTargetCells;
  XStreamMapType m_XTargetCells;

private:
  template <class T>
  Self &
  SendToTargets(const T & _arg)
  {
    /** Send input to the target c-streams. */
    for (const auto & cell : m_CTargetCells)
    {
      *(cell.second) << _arg;
    }

    /** Send input to the target xout-objects. */
    for (const auto & cell : m_XTargetCells)
    {
      *(cell.second) << _arg;
    }

    return *this;
  } // end SendToTargets
};

} // end namespace xoutlibrary

#endif // end #ifndef xoutbase_h
