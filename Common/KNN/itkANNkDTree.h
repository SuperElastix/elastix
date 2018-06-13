/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkANNkDTree_h
#define __itkANNkDTree_h

#include "itkBinaryANNTreeBase.h"

namespace itk
{

/**
 * \class ANNkDTree
 *
 * \brief
 *
 *
 * \ingroup ANNwrap
 */

template< class TListSample >
class ANNkDTree : public BinaryANNTreeBase< TListSample >
{
public:

  /** Standard itk. */
  typedef ANNkDTree                        Self;
  typedef BinaryANNTreeBase< TListSample > Superclass;
  typedef SmartPointer< Self >             Pointer;
  typedef SmartPointer< const Self >       ConstPointer;

  /** New method for creating an object using a factory. */
  itkNewMacro( Self );

  /** ITK type info. */
  itkTypeMacro( ANNkDTree, BinaryANNTreeBase );

  /** Typedef's from Superclass. */
  typedef typename Superclass::SampleType                 SampleType;
  typedef typename Superclass::MeasurementVectorType      MeasurementVectorType;
  typedef typename Superclass::MeasurementVectorSizeType  MeasurementVectorSizeType;
  typedef typename Superclass::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;

  /** Typedef's. */
  typedef ANNpointSet  ANNPointSetType;
  typedef ANNkd_tree   ANNkDTreeType;
  typedef ANNsplitRule SplittingRuleType;
  typedef unsigned int BucketSizeType;

  /** Set and get the bucket size: the number of points in a region/bucket. */
  itkSetMacro( BucketSize, BucketSizeType );
  itkGetConstMacro( BucketSize, BucketSizeType );

  /** Set and get the splitting rule: it defines how the space is divided. */
  itkSetMacro( SplittingRule, SplittingRuleType );
  itkGetConstMacro( SplittingRule, SplittingRuleType );
  void SetSplittingRule( std::string rule );

  std::string GetSplittingRule( void );

  /** Set the maximum number of points that are to be visited. */
  //void SetMaximumNumberOfPointsToVisit( unsigned int num )
  //{
  //  annMaxPtsVisit( static_cast<int>( num ) );
  //}

  /** Generate the tree. */
  virtual void GenerateTree( void );

  /** Get the ANN tree. */
  virtual ANNPointSetType * GetANNTree( void ) const
  {
    return this->m_ANNTree;
  }


protected:

  /** Constructor. */
  ANNkDTree();

  /** Destructor. */
  virtual ~ANNkDTree();

  /** PrintSelf. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Member variables. */
  ANNkDTreeType *   m_ANNTree;
  SplittingRuleType m_SplittingRule;
  BucketSizeType    m_BucketSize;

private:

  ANNkDTree( const Self & );        // purposely not implemented
  void operator=( const Self & );   // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkANNkDTree.hxx"
#endif

#endif // end #ifndef __itkANNkDTree_h
