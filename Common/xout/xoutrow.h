/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __xoutrow_h
#define __xoutrow_h

#include "xoutbase.h"
#include "xoutcell.h"
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

template< class charT, class traits = char_traits< charT > >
class xoutrow : public xoutbase< charT, traits >
{
public:

  typedef xoutrow                   Self;
  typedef xoutbase< charT, traits > Superclass;

  /** Typedefs of Superclass */
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

  /** Extra typedefs */
  typedef xoutcell< charT, traits > XOutCellType;

  /** Constructor */
  xoutrow();

  /** Destructor */
  virtual ~xoutrow();

  /** Write the buffered cell data in a row to the outputs,
   * separated by tabs.
   */
  virtual void WriteBufferedData( void );

  /** Writes the names of the target cells to the outputs;
   * This method can also be executed by selecting the
   * "WriteHeaders" target: xout["WriteHeaders"]
   */
  virtual void WriteHeaders( void );

  /** This method adds an xoutcell to the map of Targets. */
  virtual int AddTargetCell( const char * name );

  /** This method removes an xoutcell to the map of Targets. */
  virtual int RemoveTargetCell( const char * name );

  /** Method to set all targets at once. The outputs of these targets
   * are not set automatically, so make sure to do it yourself.
   */
  virtual void SetTargetCells( const XStreamMapType & cellmap );

  /** Add/Remove an output stream (like cout, or an fstream, or an xout-object).
   * In addition to the behaviour of the Superclass's methods, these functions
   * set the outputs of the TargetCells as well.
   */
  virtual int AddOutput( const char * name, ostream_type * output );

  virtual int AddOutput( const char * name, Superclass * output );

  virtual int RemoveOutput( const char * name );

  virtual void SetOutputs( const CStreamMapType & outputmap );

  virtual void SetOutputs( const XStreamMapType & outputmap );

protected:

  /** Returns a target cell.
   * Extension: if input = "WriteHeaders" it calls
   * this->WriteHeaders() and returns 'this'.
   */
  virtual Superclass & SelectXCell( const char * name );

  XStreamMapType m_CellMap;

};

} // end namespace xoutlibrary

#include "xoutrow.hxx"

#endif // end #ifndef __xoutrow_h
