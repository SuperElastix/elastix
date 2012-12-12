#ifndef ITKAFFINELOGTRANSFORM_TXX
#define ITKAFFINELOGTRANSFORM_TXX

//#include "/core/vnl/vnl_matrix_exp.h"

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
  off[0] = offset[0];
  off[1] = offset[1];
  off[2] = offset[2];
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
        this->m_Parameters[k] = this->m_Offset(j);
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
  j.Fill(0.0);

  const JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;

  /** Compute dR/dmu * (p-c) */
  const InputVectorType pp = p - this->GetCenter();
  for(unsigned int dim = 0; dim < d; dim++ )
  {
    const InputVectorType column = jsj[dim] * pp;
    for (unsigned int i = 0; i < d; ++i)
    {
      j(i,dim) = column[i];
    }
  }

//  const unsigned int blockOffset = d*d;
//  for(unsigned int dim=0; dim < d; dim++ )
// {
//    j[ dim ][ blockOffset + dim ] = 1.0;
//  }

  nzji = this->m_NonZeroJacobianIndices;

}

// Precompute Jacobian of Spatial Jacobian
template <class TScalarType, unsigned int Dimension>
void
AffineLogTransform<TScalarType, Dimension>
::PrecomputeJacobianOfSpatialJacobian( void )
{
    /** The Jacobian of spatial Jacobian is constant over inputspace, so is precomputed */
    JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
    unsigned int d = Dimension;
    unsigned int ParametersDimension = d*(d+1);

    jsj.resize(ParametersDimension);

    vnl_matrix< ScalarType > dA(d+1,d+1);
    vnl_matrix< ScalarType > A_bar(2*(d+1),2*(d+1));
    //MatrixType B_bar;//(2*(d+1),2*(d+1));
    vnl_matrix< ScalarType > B_bar(2*(d+1),2*(d+1));

    dA.fill(0.0);
    A_bar.fill(0.0);

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
                    jsj[m] = B_bar(k,l);
                }
            }

            dA.fill(0.0);
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
        jsj[m] = B_bar;
        dA.fill(0.0);
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

#endif // ITKAFFINELOGTRANSFORM_TXX
