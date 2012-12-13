/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkAffineLogTransform_txx
#define __itkAffineLogTransform_txx


#include "vnl/vnl_matrix_exp.h"
#include "itkAffineLogTransform.h"

namespace itk
{

// Constructor with default arguments
template <class TScalarType, unsigned int Dimension>
AffineLogTransform<TScalarType, Dimension>
::AffineLogTransform(const MatrixType & matrix,
                   const OutputPointType & offset)
{
  this->SetMatrix(matrix);

  OffsetType off;
  for(unsigned int i = 0; i < Dimension; i ++)
  {
	off[i] = offset[i];
  }

  this->SetOffset(off);

  // this->ComputeMatrix?
  this->PrecomputeJacobianOfSpatialJacobian();
}

// Set Parameters
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>
::SetParameters( const ParametersType & parameters )
{
   itkDebugMacro( << "Setting parameters " << parameters );

   vnl_matrix<ScalarType> exponentMatrix(this->m_MatrixLogDomain.rows(),
                                         this->m_MatrixLogDomain.cols());

  unsigned int d = Dimension;

  exponentMatrix = vnl_matrix_exp( this->m_MatrixLogDomain );

  for( unsigned int i = 0; i < d; i++)
  {
      for(unsigned int j = 0; j < d; j++)
      {
        this->m_Matrix(i,j) = exponentMatrix(i,j);
      }
  }

  for( unsigned int j = 0; j < d; j++)
  {
      this->m_Offset[j] = exponentMatrix(d+1,j);
  }

  this->SetMatrix( this->m_Matrix );
  this->SetOffset( this->m_Offset );
  
    // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.

  this->Modified();


  itkDebugMacro(<<"After setting parameters ");
}

// Compute the log domain matrix
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>
::ComputeMatrixLogDomain( void )
{
    unsigned int d = Dimension;
    unsigned int j = 0;
    vnl_matrix< ScalarType > matrix(d+1, d+1);

    for(unsigned int k = 0; k < d; k++)
    {
        for(unsigned int l = 0; l < d; l++)
        {
            matrix(k,l) = this->m_Parameters[j];
            j += 1;
        }
    }

    for( unsigned int l = 0; l < d; l++)
    {
        matrix(d+1,l) = this->m_Parameters[j];
        j += 1;
    }

    this->m_MatrixLogDomain = matrix;
}

// Get Parameters
template <class TScalarType, unsigned int Dimension>
const typename AffineLogTransform<TScalarType, Dimension>::ParametersType &
AffineLogTransform<TScalarType, Dimension>
::GetParameters( void ) const
{
    unsigned int k = 0;
    for(unsigned int i = 0; i < Dimension; i++)
    {
        for(unsigned int j = 0; j < Dimension; j++)
        {
            this->m_Parameters[k] = this->m_Matrix(i,j);
            k += 1;
        }
    }

    for(unsigned int j = 0; j < Dimension; j++)
    {
        this->m_Parameters[k] = this->m_Offset[j];
        k += 1;
    }

    return this->m_Parameters;
}

// SetIdentity()
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>
::SetIdentity( void )
{
  //Superclass::SetIdentity();
  for(unsigned int i = 0; i < this->m_Parameters.size(); i+=(Dimension+1) )
  {
      this->m_Parameters[i] = 1;
  }

  this->PrecomputeJacobianOfSpatialJacobian();
}

//Get Jacobian
template<class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::
GetJacobian( const InputPointType & p,
    JacobianType & j,
    NonZeroJacobianIndicesType & nzji) const
{
    unsigned int d = Dimension;
    unsigned int ParametersDimension = d*(d+1);
  j.SetSize(d, ParametersDimension );
  j.Fill(itk::NumericTraits<ScalarType>::Zero);

  const JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;

  const InputVectorType pp = p - this->GetCenter();
  for(unsigned int dim = 0; dim < ParametersDimension; dim++ )
  {
    //const InputVectorType column = jsj[dim] * pp;
    for (unsigned int i = 0; i < d; ++i)
    {
      j(i,dim) = pp[i];
    }
  }

 const unsigned int blockOffset = d*d;
 for(unsigned int dim=0; dim < d; dim++ )
 {
    j[ dim ][ blockOffset + dim ] = 1.0;
 }

  nzji = this->m_NonZeroJacobianIndices;

}

// Precompute Jacobian of Spatial Jacobian
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>
::PrecomputeJacobianOfSpatialJacobian( void )
{
    unsigned int d = Dimension;
    unsigned int ParametersDimension = d*(d+1);
    
    /** The Jacobian of spatial Jacobian is constant over inputspace, so is precomputed */
    JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;

    jsj.resize(ParametersDimension);

    vnl_matrix< ScalarType > dA(d+1,d+1);
    vnl_matrix< ScalarType > dummymatrix(d+1,d+1);
    vnl_matrix< ScalarType > A_bar(2*(d+1),2*(d+1));
    vnl_matrix< ScalarType > B_bar(2*(d+1),2*(d+1));

    dA.fill(itk::NumericTraits<ScalarType>::Zero);
    dummymatrix.fill(itk::NumericTraits<ScalarType>::Zero);
    A_bar.fill(itk::NumericTraits<ScalarType>::Zero);

    // Fill A_bar top left and bottom right with A
    for(unsigned int k = 0; k < d+1; k++)
    {
        for(unsigned int l = 0; l < d+1; l++)
        {
            A_bar(k,l) = this->m_MatrixLogDomain(k-d-1,l-d-1);
        }
    }
    for(unsigned int k = d+1; k < 2*(d+1); k++)
    {
        for(unsigned int l = d+1; l < 2*(d+1); l++)
        {
            A_bar(k,l) = this->m_MatrixLogDomain(k-d-1,l-d-1);
        }
    }

    unsigned int m = 0; //Dummy loop index

    //Non-translation derivatives
    for(unsigned int i = 0; i < d; i++)
    {
        for(unsigned int j = 0; j < d; j++)
        {
            dA(i,j) = 1;
            for(unsigned int k = 0; k < (d+1); k++)
            {
                for(unsigned int l = (d+1); l < 2*(d+1); l++)
                {
                    A_bar(k,l) = dA(k,(l-d-1));
                }
            }
            B_bar = vnl_matrix_exp( A_bar );
            for(unsigned int k = 0; k < (d+1); k++)
            {
                for(unsigned int l = d+1; l < 2*(d+1); l++)
                {
                    dummymatrix(k,(l-d-1)) = B_bar(k,l);
                }
            }
			jsj[m] = dummymatrix;
            dA.fill(itk::NumericTraits<ScalarType>::Zero);
            m += 1;
        }
    }

    //Translation derivatives
    for(unsigned int j = 0; j < d; j++)
    {
        dA(d+1,j) = 1;
        for(unsigned int k = 0; k < (d+1); k++)
        {
            for(unsigned int l = d+1; l < 2*(d+1); l++)
            {
                A_bar(k,l) = dA(k,(l-d-1));
            }
        }
        B_bar = vnl_matrix_exp( A_bar );
        
        for(unsigned int k = 0; k < (d+1); k++)
        {
            for(unsigned int l = d+1; l < 2*(d+1); l++)
            {
                 dummymatrix(k,(l-d-1)) = B_bar(k,l);
            }
        }
        jsj[m] = dummymatrix;
        dA.fill(itk::NumericTraits<ScalarType>::Zero);
        m += 1;
    }
}


// Print self
template<class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>::
PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "parameters:" << this->m_Parameters << std::endl;

}

} // end namespace

#endif // itkAffineLogTransform_TXX
