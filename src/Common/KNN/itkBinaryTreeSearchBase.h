/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBinaryTreeSearchBase_h
#define __itkBinaryTreeSearchBase_h

#include "itkObject.h"
#include "itkArray.h"

#include "itkBinaryTreeBase.h"

namespace itk
{

  /**
   * \class BinaryTreeSearchBase
   *
   * \brief
   *
   *
   * \ingroup ANNwrap
   */

  template < class TListSample >
  class BinaryTreeSearchBase : public Object
  {
  public:

    /** Standard itk. */
    typedef BinaryTreeSearchBase        Self;
    typedef Object                      Superclass;
    typedef SmartPointer< Self >        Pointer;
    typedef SmartPointer< const Self >  ConstPointer;

    /** ITK type info. */
    itkTypeMacro( BinaryTreeSearchBase, Object );

    /** Typedef's. */
    typedef TListSample                 ListSampleType;
    typedef BinaryTreeBase< ListSampleType > BinaryTreeType;
    typedef typename BinaryTreeType::
      MeasurementVectorType             MeasurementVectorType;
    typedef Array< int >                IndexArrayType;
    typedef Array< double >             DistanceArrayType;

    /** Set and get the binary tree. */
    virtual void SetBinaryTree( BinaryTreeType * tree );
    const BinaryTreeType * GetBinaryTree( void ) const;

    /** Set and get the number of nearest neighbours k. */
    itkSetMacro( KNearestNeighbors, unsigned int );
    itkGetConstMacro( KNearestNeighbors, unsigned int );

    /** Search the nearest neighbours of a query point qp. */
    virtual void Search( const MeasurementVectorType & qp, IndexArrayType & ind,
      DistanceArrayType & dists ) = 0;

  protected:

    BinaryTreeSearchBase();
    virtual ~BinaryTreeSearchBase();

    /** Member variables. */
    typename BinaryTreeType::Pointer    m_BinaryTree;
    unsigned int   m_KNearestNeighbors;
    unsigned int   m_DataDimension;

  private:

    BinaryTreeSearchBase( const Self& );  // purposely not implemented
    void operator=( const Self& );        // purposely not implemented

  };


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryTreeSearchBase.hxx"
#endif

#endif // end #ifndef __itkBinaryTreeSearchBase_h
