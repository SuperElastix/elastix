/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkANNbdTree_hxx
#define __itkANNbdTree_hxx

#include "itkANNbdTree.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template< class TListSample >
ANNbdTree< TListSample >
::ANNbdTree()
{
  this->m_ShrinkingRule = ANN_BD_SIMPLE;

}   // end Constructor()


/**
 * ************************ SetShrinkingRule *************************
 */

template< class TListSample >
void
ANNbdTree< TListSample >
::SetShrinkingRule( std::string rule )
{
  if( rule == "ANN_BD_NONE" )
  {
    this->m_ShrinkingRule = ANN_BD_NONE;
  }
  else if( rule == "ANN_BD_SIMPLE" )
  {
    this->m_ShrinkingRule = ANN_BD_SIMPLE;
  }
  else if( rule == "ANN_BD_CENTROID" )
  {
    this->m_ShrinkingRule = ANN_BD_CENTROID;
  }
  else if( rule == "ANN_BD_SUGGEST" )
  {
    this->m_ShrinkingRule = ANN_BD_SUGGEST;
  }
  else
  {
    itkWarningMacro( << "WARNING: No such shrinking rule." );
  }

}   // end SetShrinkingRule()


/**
 * ************************ GetShrinkingRule *************************
 */

template< class TListSample >
std::string
ANNbdTree< TListSample >
::GetShrinkingRule( void )
{
  switch( this->m_ShrinkingRule )
  {
    case ANN_BD_NONE:
      return "ANN_BD_NONE";
    case ANN_BD_SIMPLE:
      return "ANN_BD_SIMPLE";
    case ANN_BD_CENTROID:
      return "ANN_BD_CENTROID";
    case ANN_BD_SUGGEST:
      return "ANN_BD_SUGGEST";
  }

}   // end GetShrinkingRule()


/**
 * ************************ GenerateTree *************************
 */

template< class TListSample >
void
ANNbdTree< TListSample >
::GenerateTree( void )
{
  int dim = static_cast< int >( this->GetDataDimension() );
  int nop = static_cast< int >( this->GetActualNumberOfDataPoints() );
  int bcs = static_cast< int >( this->m_BucketSize );

  ANNBinaryTreeCreator::DeleteANNkDTree( this->m_ANNTree );

  this->m_ANNTree = ANNBinaryTreeCreator::CreateANNbdTree(
    this->GetSample()->GetInternalContainer(), nop, dim, bcs,
    this->m_SplittingRule, this->m_ShrinkingRule );

}   // end GenerateTree()


/**
 * ************************ PrintSelf *************************
 */

template< class TListSample >
void
ANNbdTree< TListSample >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "ShrinkingRule: " << this->m_ShrinkingRule << std::endl;

}   // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __itkANNbdTree_hxx
