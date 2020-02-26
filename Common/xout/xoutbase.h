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
#ifndef __xoutbase_h
#define __xoutbase_h

/** Get rid of warnings about too long variable names.*/
#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#pragma warning ( disable : 4503 )
#endif

#include <iostream>
#include <ostream>
#include <map>
#include <string>

namespace xoutlibrary
{
using namespace std;

/**
 * \class xoutbase
 * \brief Base class for xout.
 *
 * An abstract base class, which defines the interface
 * for using xout.
 *
 * \ingroup xout
 */

template< class charT, class traits = char_traits< charT > >
class xoutbase
{
public:

  /** Typedef's.*/
  typedef xoutbase Self;

  typedef traits                         traits_type;
  typedef charT                          char_type;
  typedef typename traits::int_type      int_type;
  typedef typename traits::pos_type      pos_type;
  typedef typename traits::off_type      off_type;
  typedef basic_ostream< charT, traits > ostream_type;
  typedef basic_ios< charT, traits >     ios_type;

  typedef std::map< std::string, ostream_type * > CStreamMapType;
  typedef std::map< std::string, Self * >         XStreamMapType;
  typedef typename CStreamMapType::iterator       CStreamMapIteratorType;
  typedef typename XStreamMapType::iterator       XStreamMapIteratorType;
  typedef typename CStreamMapType::value_type     CStreamMapEntryType;
  typedef typename XStreamMapType::value_type     XStreamMapEntryType;

  /** Constructors */
  xoutbase() = default;

  /** Destructor */
  virtual ~xoutbase() = default;

  /** The operator [] simply calls this->SelectXCell(cellname).
   * It returns an x-cell */
  inline Self & operator[]( const char * cellname );

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

  template< class T >
  Self & operator<<( const T & _arg )
  {
    return this->SendToTargets( _arg );
  }


  Self & operator<<( ostream_type & (* pf)( ostream_type  & ) )
  {
    return this->SendToTargets( pf );
  }


  Self & operator<<( ios_type & (* pf)( ios_type & ) )
  {
    return this->SendToTargets( pf );
  }


  Self & operator<<( ios_base & (* pf)( ios_base & ) )
  {
    return this->SendToTargets( pf );
  }


  virtual void WriteBufferedData( void );

  /**
   * Methods to Add and Remove target cells. They return 0 when successful.
   */
  virtual int AddTargetCell( const char * name, ostream_type * cell );

  virtual int AddTargetCell( const char * name, Self * cell );

  virtual int AddTargetCell( const char * /** name */ ){ return 1; }
  virtual int RemoveTargetCell( const char * name );

  virtual void SetTargetCells( const CStreamMapType & cellmap );

  virtual void SetTargetCells( const XStreamMapType & cellmap );

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).  */
  virtual int AddOutput( const char * name, ostream_type * output );

  virtual int AddOutput( const char * name, Self * output );

  virtual int RemoveOutput( const char * name );

  virtual void SetOutputs( const CStreamMapType & outputmap );

  virtual void SetOutputs( const XStreamMapType & outputmap );

  /** Get the output maps. */
  virtual const CStreamMapType & GetCOutputs( void );

  virtual const XStreamMapType & GetXOutputs( void );

protected:

  /** Returns a target cell. */
  virtual Self & SelectXCell( const char * name );

  /** Maps that contain the outputs. */
  CStreamMapType m_COutputs;
  XStreamMapType m_XOutputs;

  /** Maps that contain the target cells. The << operator passes its
   * input to these maps. */
  CStreamMapType m_CTargetCells;
  XStreamMapType m_XTargetCells;

  /** Boolean that says whether the Callback-function must be called.
   * False by default. */
  bool m_Call{ false };

  /** Called each time << is used, but only when m_Call == true; */
  virtual void Callback( void ){}

  template< class T >
  Self & SendToTargets( const T & _arg )
  {
    Send< T >::ToTargets( const_cast< T & >( _arg ), m_CTargetCells, m_XTargetCells );
    /** Call the callback method. */
    if( m_Call )
    {
      this->Callback();
    }
    return *this;
  } // end SendToTargets


private:

  template< class T >
  class Send
  {
public:

    static void ToTargets( T & _arg, CStreamMapType & CTargetCells, XStreamMapType & XTargetCells )
    {
      /** Send input to the target c-streams. */
      for( CStreamMapIteratorType cit = CTargetCells.begin();
        cit != CTargetCells.end(); ++cit )
      {
        *( cit->second ) << _arg;
      }

      /** Send input to the target xout-objects. */
      for( XStreamMapIteratorType xit = XTargetCells.begin();
        xit != XTargetCells.end(); ++xit )
      {
        *( xit->second ) << _arg;
      }

    } // end ToTargets


  };

};

} // end namespace xoutlibrary

#include "xoutbase.hxx"

#endif // end #ifndef __xoutbase_h
