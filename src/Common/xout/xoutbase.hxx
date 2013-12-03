/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __xoutbase_hxx
#define __xoutbase_hxx

#include "xoutbase.h"

namespace xoutlibrary
{
using namespace std;

/**
 * ********************* Constructor ****************************
 */

template< class charT, class traits >
xoutbase< charT, traits >::xoutbase()
{
  this->m_Call = false;

}   // end Constructor


/**
 * ********************* Destructor *****************************
 */

template< class charT, class traits >
xoutbase< charT, traits >::~xoutbase()
{
  //nothing

}   // end Destructor


/**
 * ********************* operator[] *****************************
 */

template< class charT, class traits >
xoutbase< charT, traits > &
xoutbase< charT, traits >::operator[]( const char * cellname )
{
  return this->SelectXCell( cellname );

}   // end operator[]


/**
 * ******************** WriteBufferedData ***********************
 *
 * This method can be overriden in inheriting classes. They
 * could for example define a specific order in which the
 * cells are flushed.
 */

template< class charT, class traits >
void
xoutbase< charT, traits >::WriteBufferedData( void )
{
  /** Update the target c-streams. */
  for( CStreamMapIteratorType cit = this->m_CTargetCells.begin();
    cit != this->m_CTargetCells.end(); ++cit )
  {
    *( cit->second ) << flush;
  }

  /** WriteBufferedData of the target xout-objects. */
  for( XStreamMapIteratorType xit = this->m_XTargetCells.begin();
    xit != this->m_XTargetCells.end(); ++xit )
  {
    ( *( xit->second ) ).WriteBufferedData();
  }

}   // end WriteBufferedData


/**
 * **************** AddTargetCell (ostream_type) ****************
 */

template< class charT, class traits >
int
xoutbase< charT, traits >::AddTargetCell( const char * name, ostream_type * cell )
{
  int returndummy = 1;

  if( this->m_XTargetCells.count( name ) )
  {
    /** an X-cell with the same name already exists */
    returndummy = 2;
  }
  else
  {
    if( this->m_CTargetCells.count( name ) == 0 )
    {
      this->m_CTargetCells.insert( CStreamMapEntryType( name, cell ) );
      returndummy = 0;
    }
  }

  return returndummy;

}   // end AddTargetCell


/**
 * **************** AddTargetCell (xoutbase) ********************
 */

template< class charT, class traits >
int
xoutbase< charT, traits >::AddTargetCell( const char * name, Self * cell )
{
  int returndummy = 1;

  if( this->m_CTargetCells.count( name ) )
  {
    /** a C-cell with the same name already exists */
    returndummy = 2;
  }
  else
  {
    if( this->m_XTargetCells.count( name ) == 0 )
    {
      this->m_XTargetCells.insert( XStreamMapEntryType( name, cell ) );
      returndummy = 0;
    }
  }

  return returndummy;

}   // end AddTargetCell


/**
 * ***************** RemoveTargetCell ***************************
 */

template< class charT, class traits >
int
xoutbase< charT, traits >::RemoveTargetCell( const char * name )
{
  int returndummy = 1;

  if( this->m_XTargetCells.count( name ) )
  {
    this->m_XTargetCells.erase( name );
    returndummy = 0;
  }

  if( this->m_CTargetCells.count( name ) )
  {
    this->m_CTargetCells.erase( name );
    returndummy = 0;
  }

  return returndummy;

}   // end RemoveTargetCell


/**
 * **************** SetTargetCells (ostream_types) **************
 */

template< class charT, class traits >
void
xoutbase< charT, traits >::SetTargetCells( const CStreamMapType & cellmap )
{
  this->m_CTargetCells = cellmap;

}   // end SetTargetCells


/**
 * **************** SetTargetCells (xout objects) ***************
 */
template< class charT, class traits >
void
xoutbase< charT, traits >::SetTargetCells( const XStreamMapType & cellmap )
{
  this->m_XTargetCells = cellmap;

}   // end SetTargetCells


/**
 * **************** AddOutput (ostream_type) ********************
 */

template< class charT, class traits >
int
xoutbase< charT, traits >::AddOutput( const char * name, ostream_type * output )
{
  int returndummy = 1;

  if( this->m_XOutputs.count( name ) )
  {
    returndummy = 2;
  }
  else
  {
    if( this->m_COutputs.count( name ) == 0 )
    {
      this->m_COutputs.insert( CStreamMapEntryType( name, output ) );
      returndummy = 0;
    }
  }

  return returndummy;

}   // end AddOutput


/**
 * **************** AddOutput (xoutbase) ************************
 */

template< class charT, class traits >
int
xoutbase< charT, traits >::AddOutput( const char * name, Self * output )
{
  int returndummy = 1;

  if( this->m_COutputs.count( name ) )
  {
    returndummy = 2;
  }
  else
  {
    if( this->m_XOutputs.count( name ) == 0 )
    {
      this->m_XOutputs.insert( XStreamMapEntryType( name, output ) );
      returndummy = 0;
    }
  }

  return returndummy;

}   // end AddOutput


/**
 * *********************** RemoveOutput *************************
 */

template< class charT, class traits >
int
xoutbase< charT, traits >::RemoveOutput( const char * name )
{
  int returndummy = 1;

  if( this->m_XOutputs.count( name ) )
  {
    this->m_XOutputs.erase( name );
    returndummy = 0;
  }

  if( this->m_COutputs.count( name ) )
  {
    this->m_COutputs.erase( name );
    returndummy = 0;
  }

  return returndummy;

}   // end RemoveOutput


/**
 * ******************* SetOutputs (ostream_types) ***************
 */

template< class charT, class traits >
void
xoutbase< charT, traits >::SetOutputs( const CStreamMapType & outputmap )
{
  this->m_COutputs = outputmap;

}   // end SetOutputs


/**
 * **************** SetOutputs (xoutobjects) ********************
 */

template< class charT, class traits >
void
xoutbase< charT, traits >::SetOutputs( const XStreamMapType & outputmap )
{
  this->m_XOutputs = outputmap;

}   // end SetOutputs


/**
 * ************************ SelectXCell *************************
 *
 * Returns a target cell.
 */

template< class charT, class traits >
xoutbase< charT, traits > &
xoutbase< charT, traits >::SelectXCell( const char * name )
{
  if( this->m_XTargetCells.count( name ) )
  {
    return *( this->m_XTargetCells[ name ] );
  }
  else
  {
    return *this;
  }

}   // end SelectXCell


/**
 * **************** GetOutputs (map of xoutobjects) *************
 */

template< class charT, class traits >
const typename xoutbase< charT, traits >::XStreamMapType
& xoutbase< charT, traits >::GetXOutputs( void )
{
  return this->m_XOutputs;

}   // end GetOutputs

/**
 * **************** GetOutputs (map of c-streams) ***************
 */

template< class charT, class traits >
const typename xoutbase< charT, traits >::CStreamMapType
& xoutbase< charT, traits >::GetCOutputs( void )
{
  return this->m_COutputs;

}   // end GetOutputs

} // end namespace xoutlibrary

#endif // end #ifndef __xoutbase_hxx
