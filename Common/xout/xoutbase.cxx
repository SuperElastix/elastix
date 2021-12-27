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

#include "xoutbase.h"

namespace xoutlibrary
{

/**
 * ********************* Destructor *****************************
 *
 * The destructor is defined here, as it is declared pure virtual.
 * (A pure virtual member function cannot be defined directly at
 * its declaration, in C++11, apparently.)
 */

xoutbase::~xoutbase() = default;


/**
 * ********************* operator[] *****************************
 */

xoutbase & xoutbase::operator[](const char * cellname)
{
  const auto found = m_XTargetCells.find(cellname);
  return (found == m_XTargetCells.end()) ? *this : *(found->second);

} // end operator[]


/**
 * ******************** WriteBufferedData ***********************
 *
 * This method can be overriden in inheriting classes. They
 * could for example define a specific order in which the
 * cells are flushed.
 */

void
xoutbase::WriteBufferedData()
{
  /** Update the target c-streams. */
  for (const auto & cell : m_CTargetCells)
  {
    *(cell.second) << std::flush;
  }

  /** WriteBufferedData of the target xout-objects. */
  for (const auto & cell : m_XTargetCells)
  {
    (*(cell.second)).WriteBufferedData();
  }

} // end WriteBufferedData


/**
 * **************** AddTargetCell (std::ostream) ****************
 */

int
xoutbase::AddTargetCell(const char * name, std::ostream * cell)
{
  int returndummy = 1;

  if (this->m_XTargetCells.count(name))
  {
    /** an X-cell with the same name already exists */
    returndummy = 2;
  }
  else
  {
    this->m_CTargetCells.insert(CStreamMapEntryType(name, cell));
    returndummy = 0;
  }

  return returndummy;

} // end AddTargetCell


/**
 * **************** AddTargetCell (xoutbase) ********************
 */

int
xoutbase::AddTargetCell(const char * name, Self * cell)
{
  int returndummy = 1;

  if (this->m_CTargetCells.count(name))
  {
    /** a C-cell with the same name already exists */
    returndummy = 2;
  }
  else
  {
    this->m_XTargetCells.insert(XStreamMapEntryType(name, cell));
    returndummy = 0;
  }

  return returndummy;

} // end AddTargetCell


/**
 * ***************** RemoveTargetCell ***************************
 */

int
xoutbase::RemoveTargetCell(const char * name)
{
  int returndummy = 1;

  if (this->m_XTargetCells.erase(name) > 0)
  {
    returndummy = 0;
  }

  if (this->m_CTargetCells.erase(name) > 0)
  {
    returndummy = 0;
  }

  return returndummy;

} // end RemoveTargetCell


/**
 * **************** SetTargetCells (std::ostreams) **************
 */

void
xoutbase::SetTargetCells(const CStreamMapType & cellmap)
{
  this->m_CTargetCells = cellmap;

} // end SetTargetCells


/**
 * **************** SetTargetCells (xout objects) ***************
 */

void
xoutbase::SetTargetCells(const XStreamMapType & cellmap)
{
  this->m_XTargetCells = cellmap;

} // end SetTargetCells


/**
 * **************** AddOutput (std::ostream) ********************
 */

int
xoutbase::AddOutput(const char * name, std::ostream * output)
{
  int returndummy = 1;

  if (this->m_XOutputs.count(name))
  {
    returndummy = 2;
  }
  else
  {
    this->m_COutputs.insert(CStreamMapEntryType(name, output));
    returndummy = 0;
  }

  return returndummy;

} // end AddOutput


/**
 * **************** AddOutput (xoutbase) ************************
 */

int
xoutbase::AddOutput(const char * name, Self * output)
{
  int returndummy = 1;

  if (this->m_COutputs.count(name))
  {
    returndummy = 2;
  }
  else
  {
    this->m_XOutputs.insert(XStreamMapEntryType(name, output));
    returndummy = 0;
  }

  return returndummy;

} // end AddOutput


/**
 * *********************** RemoveOutput *************************
 */

int
xoutbase::RemoveOutput(const char * name)
{
  int returndummy = 1;

  if (this->m_XOutputs.count(name))
  {
    this->m_XOutputs.erase(name);
    returndummy = 0;
  }

  if (this->m_COutputs.count(name))
  {
    this->m_COutputs.erase(name);
    returndummy = 0;
  }

  return returndummy;

} // end RemoveOutput


/**
 * ******************* SetOutputs (std::ostreams) ***************
 */

void
xoutbase::SetOutputs(const CStreamMapType & outputmap)
{
  this->m_COutputs = outputmap;

} // end SetOutputs


/**
 * **************** SetOutputs (xoutobjects) ********************
 */

void
xoutbase::SetOutputs(const XStreamMapType & outputmap)
{
  this->m_XOutputs = outputmap;

} // end SetOutputs


/**
 * **************** GetOutputs (map of xoutobjects) *************
 */

const xoutbase::XStreamMapType &
xoutbase::GetXOutputs()
{
  return this->m_XOutputs;

} // end GetOutputs

/**
 * **************** GetOutputs (map of c-streams) ***************
 */

const xoutbase::CStreamMapType &
xoutbase::GetCOutputs()
{
  return this->m_COutputs;

} // end GetOutputs

} // end namespace xoutlibrary
