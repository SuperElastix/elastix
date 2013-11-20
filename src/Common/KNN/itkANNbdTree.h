/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkANNbdTree_h
#define __itkANNbdTree_h

#include "itkANNkDTree.h"

namespace itk
{

  /**
   * \class ANNbdTree
   *
   * \brief
   *
   *
   * \ingroup ANNwrap
   */

  template < class TListSample >
  class ANNbdTree : public ANNkDTree< TListSample >
  {
  public:

    /** Standard itk. */
    typedef ANNbdTree                     Self;
    typedef ANNkDTree< TListSample >      Superclass;
    typedef SmartPointer< Self >          Pointer;
    typedef SmartPointer< const Self >    ConstPointer;

    /** New method for creating an object using a factory. */
    itkNewMacro( Self );

    /** ITK type info. */
    itkTypeMacro( ANNbdTree, ANNkDTree );

    /** Typedef's from Superclass. */
    typedef typename Superclass::SampleType                 SampleType;
    typedef typename Superclass::MeasurementVectorType      MeasurementVectorType;
    typedef typename Superclass::MeasurementVectorSizeType  MeasurementVectorSizeType;
    typedef typename Superclass::TotalAbsoluteFrequencyType TotalAbsoluteFrequencyType;
    typedef typename Superclass::ANNPointSetType            ANNPointSetType;
    typedef typename Superclass::ANNkDTreeType              ANNkDTreeType;
    typedef typename Superclass::SplittingRuleType          SplittingRuleType;
    typedef typename Superclass::BucketSizeType             BucketSizeType;

    typedef ANNshrinkRule      ShrinkingRuleType;

    /** Set and get the shrinking rule: it defines ... */
    itkSetMacro( ShrinkingRule, ShrinkingRuleType );
    itkGetConstMacro( ShrinkingRule, ShrinkingRuleType );
    void SetShrinkingRule( std::string rule );
    std::string GetShrinkingRule( void );

    /** Generate the tree. */
    virtual void GenerateTree( void );

  protected:

    /** Constructor. */
    ANNbdTree();

    /** Destructor. */
    virtual ~ANNbdTree() {};

    /** PrintSelf. */
    virtual void PrintSelf( std::ostream& os, Indent indent ) const;

    /** Member variables. */
    ShrinkingRuleType   m_ShrinkingRule;

  private:

    ANNbdTree( const Self& );       // purposely not implemented
    void operator=( const Self& );  // purposely not implemented

  }; // end class ANNbdTree


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkANNbdTree.hxx"
#endif

#endif // end #ifndef __itkANNbdTree_h
