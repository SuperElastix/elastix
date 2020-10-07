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
#ifndef __xoutrow_hxx
#define __xoutrow_hxx

#include "xoutrow.h"

namespace xoutlibrary
{
using namespace std;


/**
 * ******************** WriteBufferedData ***********************
 *
 * This method can be overriden in inheriting classes. They
 * could for example define a specific order in which the
 * cells are flushed.
 */

template< class charT, class traits >
void
xoutrow< charT, traits >
::WriteBufferedData( void )
{
  /** Write the cell-data to the outputs, separated by tabs. */
  XStreamMapIteratorType xit   = this->m_XTargetCells.begin();
  XStreamMapIteratorType tmpIt = xit;

  for( ++tmpIt; tmpIt != this->m_XTargetCells.end(); ++xit, ++tmpIt )
  {
    /** Write a tab to the cell */
    *( xit->second ) << "\t";

    /** And send its contents to the outputs */
    xit->second->WriteBufferedData();

    /** The cell is empty now! */
  } // end for

  /** Go to the last cell and use it to send an enter to the outputs. */
  xit->second->WriteBufferedData();
  --xit;
  *( xit->second ) << "\n";
  xit->second->WriteBufferedData();

} // end WriteBufferedData()


/**
 * ******************** AddTargetCell ***************************
 */

template< class charT, class traits >
int
xoutrow< charT, traits >
::AddTargetCell( const char * name )
{
  if( this->m_CellMap.count( name ) == 0 )
  {
    /** A new cell (type xoutcell) is created. */
    std::unique_ptr<XOutCellType> cell{ new XOutCellType };
    auto& cellReference = *cell;

    /** Set the outputs equal to the outputs of this object. */
    cell->SetOutputs( this->m_COutputs );
    cell->SetOutputs( this->m_XOutputs );

    /** Stored in a map, to make sure that later we can
     * delete all memory, assigned in this function.
     */
    this->m_CellMap.insert( std::make_pair( name, std::move(cell) ) );

    /** Add the address of the cell to the TargetCell-map. */
    return this->Superclass::AddTargetCell(name, &cellReference);
  }
  else
  {
    return 1;
  }

} // end AddTargetCell()


/**
 * ********************* RemoveTargetCell ***********************
 */

template< class charT, class traits >
int
xoutrow< charT, traits >
::RemoveTargetCell( const char * name )
{
  int returndummy = 1;

  if( this->m_XTargetCells.erase( name ) > 0 )
  {
    returndummy = 0;
  }

  if( this->m_CellMap.erase( name ) > 0 )
  {
    returndummy = 0;
  }

  return returndummy;

} // end RemoveTargetCell()


/**
 * **************** SetTargetCells (xout objects) ***************
 */

template< class charT, class traits >
void
xoutrow< charT, traits >
::SetTargetCells( const XStreamMapType & cellmap )
{
  /** Clean the this->m_CellMap (cells that are created using the
   * AddTarget(const char *) method.
   */
  this->m_CellMap.clear();

  /** Replace the TargetCellMap with the input of this function.
   * The outputs are not automatically set, because the user may
   * want to keep the outputmap that was set in the cellmap.
   */
  this->m_XTargetCells = cellmap;

} // end SetTargetCells()


/**
 * ****************** AddOutput (ostream_type) ******************
 */

template< class charT, class traits >
int
xoutrow< charT, traits >
::AddOutput( const char * name, ostream_type * output )
{
  int                    returndummy = 0;

  /** Set the output in all cells. */
  for( const auto& cell : this->m_XTargetCells )
  {
    returndummy |= cell.second->AddOutput( name, output );
  }

  /** Call the Superclass's implementation. */
  returndummy |= this->Superclass::AddOutput( name, output );
  return returndummy;

} // end AddOutput()


/**
 * ********************** AddOutput (xoutbase) ******************
 */

template< class charT, class traits >
int
xoutrow< charT, traits >
::AddOutput( const char * name, Superclass * output )
{
  int                    returndummy = 0;

  /** Set the output in all cells. */
  for( const auto& cell : this->m_XTargetCells )
  {
    returndummy |= cell.second->AddOutput( name, output );
  }

  /** Call the Superclass's implementation. */
  returndummy |= this->Superclass::AddOutput( name, output );
  return returndummy;

} // end AddOutput()


/**
 * ******************** RemoveOutput ****************************
 */

template< class charT, class traits >
int
xoutrow< charT, traits >
::RemoveOutput( const char * name )
{
  int                    returndummy = 0;
  /** Set the output in all cells. */
  for( const auto& cell : this->m_XTargetCells )
  {
    returndummy |= cell.second->RemoveOutput( name );
  }

  /** Call the Superclass's implementation. */
  returndummy |= this->Superclass::RemoveOutput( name );
  return returndummy;

} // end RemoveOutput()


/**
 * ******************* SetOutputs (ostream_types) ***************
 */

template< class charT, class traits >
void
xoutrow< charT, traits >
::SetOutputs( const CStreamMapType & outputmap )
{
  /** Set the output in all cells. */
  for( const auto& cell : this->m_XTargetCells )
  {
    cell.second->SetOutputs( outputmap );
  }

  /** Call the Superclass's implementation. */
  this->Superclass::SetOutputs( outputmap );

} // end SetOutputs()


/**
 * ******************* SetOutputs (xoutobjects) *****************
 */

template< class charT, class traits >
void
xoutrow< charT, traits >
::SetOutputs( const XStreamMapType & outputmap )
{
  /** Set the output in all cells. */
  for( const auto& cell : this->m_XTargetCells )
  {
    cell.second->SetOutputs( outputmap );
  }

  /** Call the Superclass's implementation. */
  this->Superclass::SetOutputs( outputmap );

} // end SetOutputs()


/**
 * ******************** WriteHeaders ****************************
 */

template< class charT, class traits >
void
xoutrow< charT, traits >
::WriteHeaders( void )
{
  /** Copy '*this'. */
  Self headerwriter;
  headerwriter.SetTargetCells( this->m_XTargetCells );
  // no CTargetCells, because they are not used in xoutrow!
  headerwriter.SetOutputs( this->m_COutputs );
  headerwriter.SetOutputs( this->m_XOutputs );

  /** Write the cell-names to the cells of the headerwriter. */
  for( const auto& cell : this->m_XTargetCells )
  {
    /** Write the cell's name to each cell. */
    headerwriter[ cell.first.c_str() ] << cell.first;
  } // end for
  headerwriter.WriteBufferedData();

} // end WriteHeaders()


/**
 * ********************* SelectXCell ****************************
 *
 * Returns a target cell.
 */

template< class charT, class traits >
xoutbase< charT, traits > &
xoutrow< charT, traits >
::SelectXCell( const char * name )
{
  std::string cellname( name );

  /** Check if the name is "WriteHeaders". Then the method
   * this->WriteHeaders() is invoked.
   */
  if( cellname == "WriteHeaders" )
  {
    this->WriteHeaders();
    return *this;
  }
  else
  {
    /** Call the Superclass's implementation. */
    return this->Superclass::SelectXCell( name );
  }

} // end SelectXCell()


} // end namespace xoutlibrary

#endif // end #ifndef __xoutrow_hxx
