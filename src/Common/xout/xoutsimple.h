/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
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

  template<class charT, class traits = char_traits<charT> >
    class xoutsimple : public xoutbase<charT, traits>
  {
  public:

    /** Typedef's.*/
    typedef xoutsimple                        Self;
    typedef xoutbase<charT, traits>           Superclass;

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
    virtual const CStreamMapType & GetCOutputs(void);
    virtual const XStreamMapType & GetXOutputs(void);

  };


} // end namespace xoutlibrary


#include "xoutsimple.hxx"

#endif // end #ifndef __xoutsimple_h

