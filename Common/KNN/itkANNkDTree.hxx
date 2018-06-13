/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkANNkDTree_hxx
#define __itkANNkDTree_hxx

#include "itkANNkDTree.h"
#include "itkANNBinaryTreeCreator.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template< class TListSample >
ANNkDTree< TListSample >
::ANNkDTree()
{
  this->m_ANNTree       = 0;
  this->m_SplittingRule = ANN_KD_SL_MIDPT;
  this->m_BucketSize    = 1;

}   // end Constructor()


/**
 * ************************ Destructor *************************
 */

template< class TListSample >
ANNkDTree< TListSample >
::~ANNkDTree()
{
  ANNBinaryTreeCreator::DeleteANNkDTree( this->m_ANNTree );

}   // end Destructor()


/**
 * ************************ SetSplittingRule *************************
 */

template< class TListSample >
void
ANNkDTree< TListSample >
::SetSplittingRule( std::string rule )
{
  if( rule == "ANN_KD_STD" )
  {
    this->m_SplittingRule = ANN_KD_STD;
  }
  else if( rule == "ANN_KD_MIDPT" )
  {
    this->m_SplittingRule = ANN_KD_MIDPT;
  }
  else if( rule == "ANN_KD_FAIR" )
  {
    this->m_SplittingRule = ANN_KD_FAIR;
  }
  else if( rule == "ANN_KD_SL_MIDPT" )
  {
    this->m_SplittingRule = ANN_KD_SL_MIDPT;
  }
  else if( rule == "ANN_KD_SL_FAIR" )
  {
    this->m_SplittingRule = ANN_KD_SL_FAIR;
  }
  else if( rule == "ANN_KD_SUGGEST" )
  {
    this->m_SplittingRule = ANN_KD_SUGGEST;
  }
  else
  {
    itkWarningMacro( << "WARNING: No such spliting rule." );
  }

}   // end SetSplittingRule()


/**
 * ************************ GetSplittingRule *************************
 */

template< class TListSample >
std::string
ANNkDTree< TListSample >
::GetSplittingRule( void )
{
  switch( this->m_SplittingRule )
  {
    case ANN_KD_STD:
      return "ANN_KD_STD";
    case ANN_KD_MIDPT:
      return "ANN_KD_MIDPT";
    case ANN_KD_FAIR:
      return "ANN_KD_FAIR";
    case ANN_KD_SL_MIDPT:
      return "ANN_KD_SL_MIDPT";
    case ANN_KD_SL_FAIR:
      return "ANN_KD_SL_FAIR";
    case ANN_KD_SUGGEST:
      return "ANN_KD_SUGGEST";
  }

}   // end GetSplittingRule()


/**
 * ************************ GenerateTree *************************
 */

template< class TListSample >
void
ANNkDTree< TListSample >
::GenerateTree( void )
{
  int dim = static_cast< int >( this->GetDataDimension() );
  int nop = static_cast< int >( this->GetActualNumberOfDataPoints() );
  int bcs = static_cast< int >( this->m_BucketSize );

  ANNBinaryTreeCreator::DeleteANNkDTree( this->m_ANNTree );

  this->m_ANNTree = ANNBinaryTreeCreator::CreateANNkDTree(
    this->GetSample()->GetInternalContainer(), nop, dim, bcs, this->m_SplittingRule );

}   // end GenerateTree()


/**
 * ************************ PrintSelf *************************
 */

template< class TListSample >
void
ANNkDTree< TListSample >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "ANNTree: " << this->m_ANNTree << std::endl;
  os << indent << "SplittingRule: " << this->m_SplittingRule << std::endl;
  os << indent << "BucketSize: " << this->m_BucketSize << std::endl;

}   // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __itkANNkDTree_hxx
