/*=========================================================================

Program:   Insight Segmentation & Registration Toolkit
Module:    $RCSfile: itkKernelTransform2.txx,v $
Language:  C++
Date:      $Date: 2006-11-28 14:22:18 $
Version:   $Revision: 1.1 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even 
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkKernelTransform2_txx
#define _itkKernelTransform2_txx
#include "itkKernelTransform2.h"

namespace itk
{


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  KernelTransform2<TScalarType, NDimensions>::
    KernelTransform2():Superclass(
    NDimensions, 0) 
    // the 0 is provided as
    // a tentative number for initializing the Jacobian.
    // The matrix can be resized at run time so this number
    // here is irrelevant. The correct size of the Jacobian
    // will be NDimension X NDimension.NumberOfLandMarks.
  {

    m_I.set_identity();
    m_SourceLandmarks = PointSetType::New();
    m_TargetLandmarks = PointSetType::New();
    m_Displacements   = VectorSetType::New();
    m_WMatrixComputed = false;

    m_LMatrixComputed = false;
    m_LInverseComputed = false;

    m_Stiffness = 0.0;

    // dummy value:
    this->m_PoissonRatio = 1.0;
  }

  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  KernelTransform2<TScalarType, NDimensions>::
    ~KernelTransform2()
  {
  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    SetSourceLandmarks(PointSetType * landmarks)
  {
    itkDebugMacro("setting SourceLandmarks to " << landmarks ); 
    if (this->m_SourceLandmarks != landmarks) 
    { 
      this->m_SourceLandmarks = landmarks; 
      this->UpdateParameters();
      this->Modified(); 
      // these are invalidated when the source lms change
      m_WMatrixComputed=false;
      m_LMatrixComputed=false;
      m_LInverseComputed=false;

      // you must recompute L and Linv - this does not require the targ lms
      this->ComputeLInverse();

      // precompute the nonzerojacobianindices vector
      unsigned long nrParams = this->GetNumberOfParameters();
      this->m_NonZeroJacobianIndices.resize( nrParams );
      for (unsigned int i = 0; i < nrParams; ++i )
      {
        this->m_NonZeroJacobianIndices[i] = i;
      }
    } 

    //if(m_WMatrixComputed) {
    // this->ComputeWMatrix();
    //}

  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    SetTargetLandmarks(PointSetType * landmarks)
  {
    itkDebugMacro("setting TargetLandmarks to " << landmarks ); 
    if (this->m_TargetLandmarks != landmarks) 
    { 
      this->m_TargetLandmarks = landmarks; 
      // this is invalidated when the target lms change
      m_WMatrixComputed=false;
      this->ComputeWMatrix();
      this->UpdateParameters();
      this->Modified(); 
    } 
  }


  /**
  * **************** ComputeG ***********************************
  */

  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    ComputeG( const InputVectorType &, GMatrixType & ) const
  {
    itkExceptionMacro(<< "ComputeG() should be reimplemented in the subclass !!");    
  }

  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    ComputeReflexiveG( PointsIterator, GMatrixType & GMatrix ) const
  {
    GMatrix.fill( NumericTraits< TScalarType >::Zero );
    GMatrix.fill_diagonal( m_Stiffness );    
  }


  /**
  * Default implementation of the the method. This can be overloaded
  * in transforms whose kernel produce diagonal G matrices.
  */
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    ComputeDeformationContribution( const InputPointType  & thisPoint,
    OutputPointType & result ) const
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();

    PointsIterator sp  = m_SourceLandmarks->GetPoints()->Begin();
    GMatrixType Gmatrix;

    for(unsigned int lnd=0; lnd < numberOfLandmarks; lnd++ )
    {
      ComputeG( thisPoint - sp->Value(), Gmatrix );
      for(unsigned int dim=0; dim < NDimensions; dim++ )
      {
        for(unsigned int odim=0; odim < NDimensions; odim++ )
        {
          result[ odim ] += Gmatrix(dim, odim ) * m_DMatrix(dim,lnd);
        }
      }
      ++sp;
    }

  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>
    ::ComputeD(void)
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();

    PointsIterator sp  = m_SourceLandmarks->GetPoints()->Begin();
    PointsIterator tp  = m_TargetLandmarks->GetPoints()->Begin();
    PointsIterator end = m_SourceLandmarks->GetPoints()->End();

    m_Displacements->Reserve( numberOfLandmarks );
    typename VectorSetType::Iterator vt = m_Displacements->Begin();

    while( sp != end )
    {
      vt->Value() = tp->Value() - sp->Value();
      vt++;
      sp++;
      tp++;
    }
    //	std::cout<<" Computed displacements "<<m_Displacements<<std::endl;
  }

  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>
    ::ComputeWMatrix(void)
  {
    //	std::cout<<"Computing W matrix"<<std::endl;

    typedef vnl_svd<TScalarType>  SVDSolverType;
    if(!m_LMatrixComputed) {
      this->ComputeL();
    }
    this->ComputeY();
    SVDSolverType svd( m_LMatrix, 1e-8 );
    m_WMatrix = svd.solve( m_YMatrix );

    this->ReorganizeW();
    m_WMatrixComputed=true;
  }

  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>::
    ComputeLInverse(void)
  {
    // Assumes that L has already been computed
    // Necessary for the jacobian
    if(!m_LMatrixComputed) {
      this->ComputeL();
    }
    //std::cout<<"LMatrix is:"<<std::endl;
    //std::cout<<m_LMatrix<<std::endl;
    m_LMatrixInverse=vnl_matrix_inverse<TScalarType> (m_LMatrix);
    m_LInverseComputed=true;
    //std::cout<<"LMatrix inverse is:"<<std::endl;
    //std::cout<<m_LMatrixInverse<<std::endl;
  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>::
    ComputeL(void)
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();
    vnl_matrix<TScalarType> O2(NDimensions*(NDimensions+1),
      NDimensions*(NDimensions+1), 0);

    this->ComputeP();
    this->ComputeK();

    m_LMatrix.set_size( NDimensions*(numberOfLandmarks+NDimensions+1),
      NDimensions*(numberOfLandmarks+NDimensions+1) );
    m_LMatrix.fill( 0.0 );

    m_LMatrix.update( m_KMatrix, 0, 0 );
    m_LMatrix.update( m_PMatrix, 0, m_KMatrix.columns() );
    m_LMatrix.update( m_PMatrix.transpose(), m_KMatrix.rows(), 0);
    m_LMatrix.update( O2, m_KMatrix.rows(), m_KMatrix.columns());
    m_LMatrixComputed=1;

  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>::
    ComputeK(void)
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();
    GMatrixType G;


    m_KMatrix.set_size( NDimensions * numberOfLandmarks,
      NDimensions * numberOfLandmarks );

    m_KMatrix.fill( 0.0 );

    PointsIterator p1  = m_SourceLandmarks->GetPoints()->Begin();
    PointsIterator end = m_SourceLandmarks->GetPoints()->End();

    // K matrix is symmetric, so only evaluate the upper triangle and
    // store the values in bot the upper and lower triangle
    unsigned int i = 0;
    while( p1 != end )
    {
      PointsIterator p2 = p1; // start at the diagonal element
      unsigned int j = i;

      // Compute the block diagonal element, i.e. kernel for pi->pi
      ComputeReflexiveG(p1, G);
      m_KMatrix.update(G, i*NDimensions, i*NDimensions);
      p2++;
      j++;

      // Compute the upper (and copy into lower) triangular part of K
      while( p2 != end ) 
      {
        const InputVectorType s = p1.Value() - p2.Value();
        ComputeG(s, G);
        // write value in upper and lower triangle of matrix
        m_KMatrix.update(G, i*NDimensions, j*NDimensions);
        m_KMatrix.update(G, j*NDimensions, i*NDimensions);  
        p2++;
        j++;
      }
      p1++;
      i++;
    }
    //std::cout<<"K matrix: "<<std::endl;
    //std::cout<<m_KMatrix<<std::endl;
  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>::
    ComputeP()
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();
    IMatrixType I;
    IMatrixType temp;
    InputPointType p;

    I.set_identity();
    m_PMatrix.set_size( NDimensions*numberOfLandmarks,
      NDimensions*(NDimensions+1) );
    m_PMatrix.fill( 0.0 );
    for (unsigned int i = 0; i < numberOfLandmarks; i++)
    {
      m_SourceLandmarks->GetPoint(i, &p);
      for (unsigned int j = 0; j < NDimensions; j++)
      {
        temp = I * p[j];
        m_PMatrix.update(temp, i*NDimensions, j*NDimensions);
      }
      m_PMatrix.update(I, i*NDimensions, NDimensions*NDimensions);
    }
  }



  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void KernelTransform2<TScalarType, NDimensions>::
    ComputeY(void)
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();

    this->ComputeD();

    typename VectorSetType::ConstIterator displacement =
      m_Displacements->Begin();

    m_YMatrix.set_size( NDimensions*(numberOfLandmarks+NDimensions+1), 1);

    m_YMatrix.fill( 0.0 );

    for (unsigned int i = 0; i < numberOfLandmarks; i++)
    {
      for (unsigned int j = 0; j < NDimensions; j++)
      {
        m_YMatrix.put(i*NDimensions+j, 0, displacement.Value()[j]);
      }
      displacement++;
    }

    for (unsigned int i = 0; i < NDimensions*(NDimensions+1); i++) 
    {
      m_YMatrix.put(numberOfLandmarks*NDimensions+i, 0, 0);
    }
  }


  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>
    ::ReorganizeW(void)
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();

    // The deformable (non-affine) part of the registration goes here
    m_DMatrix.set_size(NDimensions,numberOfLandmarks);
    unsigned int ci = 0;
    for(unsigned int lnd=0; lnd < numberOfLandmarks; lnd++ )
    {
      for(unsigned int dim=0; dim < NDimensions; dim++ )
      {
        m_DMatrix(dim,lnd) = m_WMatrix(ci++,0);
      }
    }

    // This matrix holds the rotational part of the Affine component
    for(unsigned int j=0; j < NDimensions; j++ )
    {
      for(unsigned int i=0; i < NDimensions; i++ )
      {
        m_AMatrix(i,j) = m_WMatrix(ci++,0);
      }
    }

    // This vector holds the translational part of the Affine component
    for(unsigned int k=0; k < NDimensions; k++ )
    {
      m_BVector(k) = m_WMatrix(ci++,0);
    }

    // release WMatrix memory by assigning a small one.
    m_WMatrix = WMatrixType(1,1);   

    m_WMatrixComputed=true;
  }



  /**
  *
  */
  template <class TScalarType, unsigned int NDimensions>
  typename KernelTransform2<TScalarType, NDimensions>::OutputPointType
    KernelTransform2<TScalarType, NDimensions>
    ::TransformPoint(const InputPointType& thisPoint) const
  {

    OutputPointType result;

    typedef typename OutputPointType::ValueType ValueType;

    result.Fill( NumericTraits< ValueType >::Zero );

    this->ComputeDeformationContribution( thisPoint, result );
    
    // Add the rotational part of the Affine component
    for(unsigned int j=0; j < NDimensions; j++ )
    {
      for(unsigned int i=0; i < NDimensions; i++ )
      {
        result[i] += m_AMatrix(i,j) * thisPoint[j];
      }
    }

    // This vector holds the translational part of the Affine component
    for(unsigned int k=0; k < NDimensions; k++ )
    {
      result[k] += m_BVector(k) + thisPoint[k];
    }

    return result;

  }




  // Compute the Jacobian in one position 
  template <class TScalarType, unsigned int NDimensions>
  const typename KernelTransform2<TScalarType,NDimensions>::JacobianType & 
    KernelTransform2< TScalarType,NDimensions>::
    GetJacobian( const InputPointType & thisPoint) const
  {     
    this->GetJacobian( thisPoint, this->m_Jacobian, this->m_NonZeroJacobianIndicesTemp );
    return m_Jacobian;
  }

  // Set to the identity transform - ie make the  Source and target lm the same
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    SetIdentity()
  {
    this->SetParameters(this->GetFixedParameters());
  }

  // Set the parameters
  // NOTE that in this transformation both the Source and Target
  // landmarks could be considered as parameters. It is assumed
  // here that the Target landmarks are provided by the user and
  // are not changed during the optimization process required for
  // registration.
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    SetParameters( const ParametersType & parameters )
  {
    //	std::cout<<"Setting parameters to "<<parameters<<std::endl;
    this->m_Parameters = parameters;
    typename PointsContainer::Pointer landmarks = PointsContainer::New();
    const unsigned int numberOfLandmarks =  parameters.Size() / NDimensions; 
    landmarks->Reserve( numberOfLandmarks );

    PointsIterator itr = landmarks->Begin();
    PointsIterator end = landmarks->End();

    InputPointType  landMark; 

    unsigned int pcounter = 0;
    while( itr != end )
    {
      for(unsigned int dim=0; dim<NDimensions; dim++)
      {
        landMark[ dim ] = parameters[ pcounter ];
        pcounter++;
      }  
      itr.Value() = landMark;
      itr++;
    }

    // m_SourceLandmarks->SetPoints( landmarks );
    m_TargetLandmarks->SetPoints( landmarks );

    // W MUST be recomputed if the target lms are set
    this->ComputeWMatrix();

    //  if(!m_LInverseComputed) {
    //  this->ComputeLInverse();
    //  }

    // Modified is always called since we just have a pointer to the
    // parameters and cannot know if the parameters have changed.
    this->Modified();

  }

  // Set the fixed parameters
  // Since the API of the SetParameters() function sets the
  // source landmarks, this function was added to support the
  // setting of the target landmarks, and allowing the Transform
  // I/O mechanism to be supported.
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    SetFixedParameters( const ParametersType & parameters )
  {
    typename PointsContainer::Pointer landmarks = PointsContainer::New();
    const unsigned int numberOfLandmarks =  parameters.Size() / NDimensions; 

    landmarks->Reserve( numberOfLandmarks );

    PointsIterator itr = landmarks->Begin();
    PointsIterator end = landmarks->End();

    InputPointType  landMark; 

    unsigned int pcounter = 0;
    while( itr != end )
    {
      for(unsigned int dim=0; dim<NDimensions; dim++)
      {
        landMark[ dim ] = parameters[ pcounter ];
        pcounter++;
      }  
      itr.Value() = landMark;
      itr++;
    }

    //  m_TargetLandmarks->SetPoints( landmarks );  
    m_SourceLandmarks->SetPoints( landmarks );  

    // these are invalidated when the source lms change
    m_WMatrixComputed=false;
    m_LMatrixComputed=false;
    m_LInverseComputed=false;

    // you must recompute L and Linv - this does not require the targ lms
    this->ComputeLInverse();

  }


  // Update parameters array
  // They are the components of all the landmarks in the source space
  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    UpdateParameters( void )
  {
    this->m_Parameters = ParametersType( m_TargetLandmarks->GetNumberOfPoints() * NDimensions );

    PointsIterator itr = m_TargetLandmarks->GetPoints()->Begin();
    PointsIterator end = m_TargetLandmarks->GetPoints()->End();

    unsigned int pcounter = 0;
    while( itr != end )
    {
      InputPointType  landmark = itr.Value();
      for(unsigned int dim=0; dim<NDimensions; dim++)
      {
        this->m_Parameters[ pcounter ] = landmark[ dim ];
        pcounter++;
      }  
      itr++;
    }
  }




  // Get the parameters
  // They are the components of all the landmarks in the source space
  template <class TScalarType, unsigned int NDimensions>
  const typename KernelTransform2<TScalarType, NDimensions>::ParametersType &
    KernelTransform2<TScalarType, NDimensions>::
    GetParameters( void ) const
  {
    //this->UpdateParameters();
    return this->m_Parameters;

  }


  // Get the fixed parameters
  // This returns the target landmark locations 
  // This was added to support the Transform Reader/Writer mechanism 
  template <class TScalarType, unsigned int NDimensions>
  const typename KernelTransform2<TScalarType, NDimensions>::ParametersType &
    KernelTransform2<TScalarType, NDimensions>::
    GetFixedParameters( void ) const
  {
    this->m_FixedParameters = ParametersType( m_SourceLandmarks->GetNumberOfPoints() * NDimensions );

    PointsIterator itr = m_SourceLandmarks->GetPoints()->Begin();
    PointsIterator end = m_SourceLandmarks->GetPoints()->End();

    unsigned int pcounter = 0;
    while( itr != end )
    {
      InputPointType  landmark = itr.Value();
      for(unsigned int dim=0; dim<NDimensions; dim++)
      {
        this->m_FixedParameters[ pcounter ] = landmark[ dim ];
        pcounter++;
      }  
      itr++;
    }

    return this->m_FixedParameters;

  }

  /**
  * ********************* GetJacobian ****************************
  */

  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    GetJacobian(
    const InputPointType & p,
    JacobianType & j,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
  {
    const unsigned long numberOfLandmarks = m_SourceLandmarks->GetNumberOfPoints();
    j.SetSize(NDimensions, numberOfLandmarks*NDimensions);
    j.Fill( 0.0 );
    GMatrixType Gmatrix;

    PointsIterator sp  = m_SourceLandmarks->GetPoints()->Begin();
    for(unsigned int lnd=0; lnd < numberOfLandmarks; lnd++ )
    {
      ComputeG( p - sp->Value(), Gmatrix );
      ///std::cout<<"G for landmark "<<lnd<<std::endl<<Gmatrix<<std::endl;
      for(unsigned int dim=0; dim < NDimensions; dim++ )
      {
        for(unsigned int odim=0; odim < NDimensions; odim++ )
        {
          for(unsigned int lidx=0; lidx < numberOfLandmarks*NDimensions; lidx++ )
          {
            j[ odim ] [lidx] += Gmatrix(dim, odim ) * 
              m_LMatrixInverse[lnd*NDimensions+dim][lidx];
          }
        }
      }
      ++sp;

    }
    for(unsigned int odim=0; odim < NDimensions; odim++ )
    {
      for(unsigned int lidx=0; lidx < numberOfLandmarks*NDimensions; lidx++ )
      {
        for(unsigned int dim=0; dim < NDimensions; dim++ )
        {
          j[ odim ] [lidx] += p[dim] * 
            m_LMatrixInverse[(numberOfLandmarks+dim)*NDimensions+odim][lidx];
        }

        j[ odim ] [lidx] += m_LMatrixInverse[(numberOfLandmarks+NDimensions)*NDimensions+odim][lidx];
      }
    }	
    
    nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;

  } // end GetJacobian()


  template <class TScalarType, unsigned int NDimensions>
  void
    KernelTransform2<TScalarType, NDimensions>::
    PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf(os,indent);
    if (m_SourceLandmarks)
    {
      os << indent << "SourceLandmarks: " << std::endl;
      m_SourceLandmarks->Print(os,indent.GetNextIndent());
    }
    if (m_TargetLandmarks)
    {
      os << indent << "TargetLandmarks: " << std::endl;
      m_TargetLandmarks->Print(os,indent.GetNextIndent());
    }
    if (m_Displacements)
    {
      os << indent << "Displacements: " << std::endl;
      m_Displacements->Print(os,indent.GetNextIndent());
    }
    os << indent << "Stiffness: " << m_Stiffness << std::endl;
  }
} // namespace itk

#endif
