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
#ifndef __xoutsimple_h
#define __xoutsimple_h

#include "xoutbase.h"

namespace xoutlibrary
{
using namespace std;

/**
 * \class xoutsimple
 * \brief xout class with only basic functionality.
 *
 * The xoutsimple class just immediately prints to the desired outputs.
 *
 * \ingroup xout
 */

template< class charT, class traits = char_traits< charT > >
class xoutsimple : public xoutbase< charT, traits >
{
public:

  /** Typedef's.*/
  typedef xoutsimple                Self;
  typedef xoutbase< charT, traits > Superclass;

  typedef typename Superclass::traits_type  traits_type;
  typedef typename Superclass::char_type    char_type;
  typedef typename Superclass::int_type     int_type;
  typedef typename Superclass::pos_type     pos_type;
  typedef typename Superclass::off_type     off_type;
  typedef typename Superclass::ostream_type ostream_type;
  typedef typename Superclass::ios_type     ios_type;

  typedef typename Superclass::CStreamMapType         CStreamMapType;
  typedef typename Superclass::XStreamMapType         XStreamMapType;
  typedef typename Superclass::CStreamMapIteratorType CStreamMapIteratorType;
  typedef typename Superclass::XStreamMapIteratorType XStreamMapIteratorType;
  typedef typename Superclass::CStreamMapEntryType    CStreamMapEntryType;
  typedef typename Superclass::XStreamMapEntryType    XStreamMapEntryType;

  /** Constructors */
  xoutsimple();

  /** Destructor */
  virtual ~xoutsimple();

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).  */
  virtual int AddOutput( const char * name, ostream_type * output );

  virtual int AddOutput( const char * name, Superclass * output );

  virtual int RemoveOutput( const char * name );

  virtual void SetOutputs( const CStreamMapType & outputmap );

  virtual void SetOutputs( const XStreamMapType & outputmap );

  /** Get the output maps. */
  virtual const CStreamMapType & GetCOutputs( void );

  virtual const XStreamMapType & GetXOutputs( void );

};

} // end namespace xoutlibrary

#include "xoutsimple.hxx"

#endif // end #ifndef __xoutsimple_h
